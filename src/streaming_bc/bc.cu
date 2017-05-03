#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "operators.cuh"
#include "algs.cuh"
#include "streaming_bc/bc.cuh"
#include "streaming_bc/bc_tree.cuh"


using namespace std;

namespace cuStingerAlgs {


void StreamingBC::Init(cuStinger& custing)
{
	forest = createBcForest(custing.nv, nr);

	if (nr == custing.nv) {
		approx = false;
	}

	for (length_t k = 0; k < nr; k++)
	{
		forest->trees_h[k]->queue.Init(custing.nv);

		// Set the root for each tree
		if (approx) {
			forest->trees_h[k]->root = rand() % forest->nv;
		} else {
			forest->trees_h[k]->root = k;
		}
	}

	host_deltas = new float[custing.nv];
	cusLB = new cusLoadBalance(custing.nv);

	// Keep a device copy of the array forest->trees_d
	trees_d = (bcTree**) allocDeviceArray(nr, sizeof(bcTree*));
	copyArrayHostToDevice(forest->trees_d, trees_d, nr, sizeof(bcTree*));

	Reset();
}


void StreamingBC::Reset()
{
	for (length_t k = 0; k < nr; k++)
	{
		bcTree *hostBcTree = forest->trees_h[k];
		hostBcTree->queue.resetQueue();
		hostBcTree->currLevel = 0;

		// initialize all offsets to zero
		for (int i = 0; i < hostBcTree->nv; i++)
		{
			hostBcTree->offsets[i] = 0;
		}

		SyncDeviceWithHost(k);
	}
}

// Must pass in a root node vertex id, and a pointer to bc values (of length custing.nv)
void StreamingBC::setInputParameters(float *bc_array)
{
	bc = bc_array;
}

void StreamingBC::Release()
{
	delete cusLB;
	delete[] host_deltas;
	destroyBcForest(forest, nr);
	freeDeviceArray(trees_d);
}


void StreamingBC::Run(cuStinger& custing)
{
	for (length_t k = 0; k < nr; k++)
	{
		RunBfsTraversal(custing, k);
		DependencyAccumulation(custing, k);
	}
}


void StreamingBC::RunBfsTraversal(cuStinger& custing, length_t k)
{
	bcTree *deviceBcTree = forest->trees_d[k];
	bcTree *hostBcTree = forest->trees_h[k];

	// Clear out array values first
	allVinG_TraverseVertices<bcOperator::setupArrays>(custing, deviceBcTree);
	hostBcTree->queue.enqueueFromHost(hostBcTree->root);
	SyncDeviceWithHost(k);

	// set d[root] <- 0
	int zero = 0;
	copyArrayHostToDevice(&zero, hostBcTree->d + hostBcTree->root,
		1, sizeof(length_t));

	// set sigma[root] <- 1
	int one = 1;
	copyArrayHostToDevice(&one, hostBcTree->sigma + hostBcTree->root,
		1, sizeof(length_t));

	length_t prevEnd = 1;
	hostBcTree->offsets[0] = 1;

	while(hostBcTree->queue.getActiveQueueSize() > 0)
	{
		allVinA_TraverseEdges_LB<bcOperator::bcExpandFrontier>(custing,
			deviceBcTree, *cusLB, hostBcTree->queue);
		SyncHostWithDevice(k);

		// Update cumulative offsets from start of queue
		hostBcTree->queue.setQueueCurr(prevEnd);

		vertexId_t level = getLevel(k);
		hostBcTree->offsets[level + 1] = hostBcTree->queue.getActiveQueueSize() + hostBcTree->offsets[level];

		prevEnd = hostBcTree->queue.getQueueEnd();

		hostBcTree->currLevel++;
		SyncDeviceWithHost(k);
	}
}


void StreamingBC::DependencyAccumulation(cuStinger& custing, length_t k)
{
	bcTree *deviceBcTree = forest->trees_d[k];
	bcTree *hostBcTree = forest->trees_h[k];

	// Iterate backwards through depths, starting from 2nd deepest frontier
	// Begin with the 2nd deepest frontier as the active queue
	hostBcTree->currLevel -= 2;
	SyncDeviceWithHost(k);

	while (getLevel(k) >= 0) {
		length_t start = hostBcTree->offsets[getLevel(k)];
		length_t end = hostBcTree->offsets[getLevel(k) + 1];

		// set queue start and end so the queue holds all nodes in one frontier
		hostBcTree->queue.setQueueCurr(start);
		hostBcTree->queue.setQueueEnd(end);
		hostBcTree->queue.SyncDeviceWithHost();
		SyncDeviceWithHost(k);

		// Now, run the macro for all outbound edges over this queue
		allVinA_TraverseEdges_LB<bcOperator::dependencyAccumulation>(custing,
			deviceBcTree, *cusLB, hostBcTree->queue);
		SyncHostWithDevice(k);

		hostBcTree->currLevel -= 1;
		SyncDeviceWithHost(k);
	}

	// Now, copy over delta values to host
	copyArrayDeviceToHost(hostBcTree->delta, host_deltas, hostBcTree->nv, sizeof(float));

	// Finally, update the bc values
	for (vertexId_t w = 0; w < hostBcTree->nv; w++) {
		if (w != hostBcTree->root) {
			bc[w] += host_deltas[w];
		}
	}
}

// TODO: Remove. This is brute force
// void StreamingBC::InsertEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
// {
// 	// vertexId_t *diffs_h = new vertexId_t[nr];
// 	// getDepthDifferences(custing, src, dst, nr, diffs_h, trees_d);

// 	// SyncHostWithDevice();  // copy all ulow, uhigh assignments to host

// 	// the count of adjacent cases and nonadjacent cases
// 	// vertexId_t adj = 0;
// 	// vertexId_t nonadj = 0;
// 	// vertexId_t size;  // the overall size of the case array

// 	vertexId_t* caseArray_h = new vertexId_t[nr];
// 	for (int i = 0; i < nr; i++) {
// 		caseArray_h[i] = i;
// 	}

// 	// Run original BFS and DA on these nonadj case roots
	
// 	// Insert edge into custing
// 	length_t allocs;
// 	BatchUpdateData bud(1 , true, custing.nv);
// 	vertexId_t *srcs = bud.getSrc();
// 	vertexId_t *dsts = bud.getDst();
// 	srcs[0] = src;
// 	dsts[0] = dst;
	
// 	// srcs[1] = dst;
// 	// dsts[1] = src;

// 	BatchUpdate bu = BatchUpdate(bud);
// 	custing.edgeInsertions(bu, allocs);

// 	printf("Done adding (%d, %d) into custing\n", src, dst);

// 	for (int i = 0; i < nr; i++) {
// 		vertexId_t k = caseArray_h[i];

// 		forest->trees_h[k]->queue.resetQueue();
// 		forest->trees_h[k]->currLevel = 0;

// 		// TODO: Remove delta of this tree from the bc values
// 		copyArrayDeviceToHost(forest->trees_h[k]->delta, host_deltas,
// 			custing.nv, sizeof(float));

// 		for (int j = 0; j < custing.nv; j++) {
// 			if (j != forest->trees_h[k]->root) {
// 				bc[j] -= host_deltas[j];
// 			}
// 		}
		
// 		printf("K: %d\tidx: %d\n", k, i);
// 		RunBfsTraversal(custing, k);
// 		printf("Done BFS traversal\n");
// 		DependencyAccumulation(custing, k);
// 		printf("Done DA\n");
// 	}

// 	// insertionNonadj(custing, caseArray_d + adj, nonadj, trees_d);

// 	// freeDeviceArray(caseArray_d);
// 	// delete[] diffs_h;
// 	delete[] caseArray_h;
// }

void StreamingBC::InsertEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
{
	// Insert edge into custing
	length_t allocs;
	BatchUpdateData bud(1 , true, custing.nv);
	vertexId_t *srcs = bud.getSrc();
	vertexId_t *dsts = bud.getDst();
	srcs[0] = src;
	dsts[0] = dst;

	BatchUpdate bu = BatchUpdate(bud);
	custing.edgeInsertions(bu, allocs);
	printf("Done adding (%d, %d) into custing\n", src, dst);

	// Now, handle the adj and nonadj cases
	vertexId_t *diffs_h = new vertexId_t[nr];
	getDepthDifferences(custing, src, dst, nr, diffs_h, trees_d);

	SyncHostWithDevice();  // copy all ulow, uhigh assignments to host

	// the count of adjacent cases and nonadjacent cases
	vertexId_t adj;
	vertexId_t nonadj;
	vertexId_t size;  // the overall size of the case array

	vertexId_t* caseArray_h = buildCaseArray(diffs_h, nr, size, adj, nonadj);
	vertexId_t* caseArray_d = (vertexId_t*) allocDeviceArray(size,
		sizeof(vertexId_t));
	copyArrayHostToDevice(caseArray_h, caseArray_d, size, sizeof(vertexId_t));
	
	// handle adjacent cases
	insertionAdj(custing, caseArray_h, caseArray_d, adj);

	// Run original BFS and DA on these nonadj case roots
	vertexId_t k;
	for (int i = 0; i < nonadj; i++) {
		k = caseArray_h[adj + i];

		forest->trees_h[k]->queue.resetQueue();
		forest->trees_h[k]->currLevel = 0;

		// Remove delta of this tree from the bc values
		copyArrayDeviceToHost(forest->trees_h[k]->delta, host_deltas,
			custing.nv, sizeof(float));

		for (int j = 0; j < custing.nv; j++) {
			if (j != forest->trees_h[k]->root) {
				bc[j] -= host_deltas[j];
			}
		}

		// Now run static bc
		printf("K: %d\tidx: %d\n", k, adj+i);
		RunBfsTraversal(custing, k);
		printf("Done BFS traversal\n");
		DependencyAccumulation(custing, k);
		printf("Done DA\n");
	}

	// insertionNonadj(custing, caseArray_d + adj, nonadj, trees_d);

	freeDeviceArray(caseArray_d);
	delete[] diffs_h;
	delete[] caseArray_h;
}


void StreamingBC::RemoveEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
{

}


void StreamingBC::getDepthDifferences(cuStinger& custing, vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_h, bcTree** trees_d)
{
	// TODO: optimize these device allocations
	depthDiffs *dDiffs_h = new depthDiffs;
	depthDiffs *dDiffs_d = (depthDiffs*) allocDeviceArray(1, sizeof(depthDiffs));

	dDiffs_h->trees_d = trees_d;
	dDiffs_h->numRoots = numRoots;
	dDiffs_h->src = src;
	dDiffs_h->dst = dst;

	// allocate space on device for diffs_d
	dDiffs_h->diffs_d = (vertexId_t*) allocDeviceArray(numRoots, sizeof(vertexId_t));
	// copy contents of host struct onto device struct
	copyArrayHostToDevice(dDiffs_h, dDiffs_d, 1, sizeof(depthDiffs));

	// we need an array of "roots" that we can use with our operator
	vertexId_t* rootArray_h = new vertexId_t[numRoots];
	for (int k = 0; k < numRoots; k++) {
		rootArray_h[k] = k;
	}

	// need the same array on the device
	vertexId_t* rootArray_d = (vertexId_t*) allocDeviceArray(numRoots,
		sizeof(vertexId_t));

	copyArrayHostToDevice(rootArray_h, rootArray_d, numRoots,
		sizeof(vertexId_t));

	// now, use an operator to get the depth differences and 
	// ulow and uhigh assignments
	allVinA_TraverseVertices<bcOperator::preprocessEdge>(custing,
		(void*) dDiffs_d, rootArray_d, numRoots);

	// store the results in diffs_h
	copyArrayDeviceToHost(dDiffs_h->diffs_d, diffs_h, numRoots,
		sizeof(vertexId_t));

	// Free device memory
	freeDeviceArray(rootArray_d);
	freeDeviceArray(dDiffs_d);
	freeDeviceArray(dDiffs_h->diffs_d);

	// Free host memory
	delete[] rootArray_h;
	delete dDiffs_h;

	SyncHostWithDevice();  // copy all ulow, uhigh assignments to host
}


vertexId_t* buildCaseArray(vertexId_t* diffs_h, length_t numRoots,
	vertexId_t& size, vertexId_t& adj, vertexId_t& nonadj)
{
	adj = 0;
	nonadj = 0;
	for (int k = 0; k < numRoots; k++) {
		if (abs(diffs_h[k]) == 1) {
			adj++;
		} else if (abs(diffs_h[k]) > 1) {
			nonadj++;
		}
	}

	size = adj + nonadj;  // the size of the case array

	// positions in the array of where to place each case type
	vertexId_t posAdj = 0;  // index used for adjacent cases
	vertexId_t posNonadj = adj;  // index used for nonadjacent cases

	vertexId_t* caseArray = new vertexId_t[size];

	for (int k = 0; k < numRoots; k++) {
		if (abs(diffs_h[k]) == 1) {
			caseArray[posAdj++] = k;
		} else if (abs(diffs_h[k]) > 1) {
			caseArray[posNonadj++] = k;
		}
	}
	return caseArray;
}

void StreamingBC::insertionAdj(cuStinger& custing, vertexId_t* adjRoots_h,
	vertexId_t* adjRoots_d, vertexId_t adjCount)
{

	printf("insertionAdj begin\n");
	adjInsertData *data_h = new adjInsertData;
	// adjInsertData *ptr_data_h = &data_h;
	data_h->trees_d = trees_d;
	data_h->numRoots = forest->numRoots;
	data_h->levelQueue.Init(custing.nv + 1);
	data_h->bfsQueue.Init(custing.nv + 1);

	data_h->frontierOffsets = new vertexId_t[custing.nv];

	// TODO: optimize these allocations
	data_h->t =  (char*) allocDeviceArray(custing.nv, sizeof(char));
	data_h->dP = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	data_h->sigmaHat = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	data_h->deltaHat = (float*) allocDeviceArray(custing.nv, sizeof(float));

	// bcVals is a device copy of bc values
	data_h->bcVals = (float*) allocDeviceArray(custing.nv, sizeof(float));
	copyArrayHostToDevice(bc, data_h->bcVals, custing.nv, sizeof(float));

	// copy d[ulow] into data_h->ulow_depth
	copyArrayDeviceToHost(tree->d + tree->ulow, &(data_h->ulow_depth), 1, sizeof(vertexId_t));

	adjInsertData* data_d = (adjInsertData*) allocDeviceArray(1,
		sizeof(adjInsertData));
	copyArrayHostToDevice(data_h, data_d, 1, sizeof(adjInsertData));

	// run algorithm sequentially for each adjacent-case tree
	for (int k = 0; k < adjCount; k++) {
		printf("Loop begin\n");
		// Stage 1: Local initialization
		data_h->levelQueue.resetQueue();
		data_h->bfsQueue.resetQueue();
		
		data_h->levelQueue.SyncDeviceWithHost();
		data_h->bfsQueue.SyncDeviceWithHost();

		// Copy over queue changes
		copyArrayHostToDevice(data_h, data_d, 1, sizeof(adjInsertData));
		vertexId_t treeIdx = adjRoots_h[k];

		allVinA_TraverseVertices<bcOperator::insertionAdjInit>(custing,
			(void*) data_d, adjRoots_d + k, 1);
		
		// Want to sync data_h->treeIdx with device copy
		copyArrayDeviceToHost(data_d, data_h, 1, sizeof(adjInsertData));
		data_h->levelQueue.SyncHostWithDevice();
		data_h->bfsQueue.SyncHostWithDevice();

		// Stage 2: BFS Traversal starting at ulow
		insertionAdjRunBFS(custing, data_h, data_d, treeIdx);
		// Stage 3: Dependency accumulation
		insertionAdjRunDA(custing, data_h, data_d, treeIdx);
	}

	// copy bcvals onto host
	copyArrayDeviceToHost(data_h->bcVals, bc, custing.nv, sizeof(float));

	freeDeviceArray(data_d);
	freeDeviceArray(data_h->t);
	freeDeviceArray(data_h->dP);
	freeDeviceArray(data_h->sigmaHat);
	freeDeviceArray(data_h->deltaHat);
	freeDeviceArray(data_h->bcVals);
	printf("Finished adj cases\n");

	delete[] data_h->frontierOffsets;
	delete data_h;
}

void StreamingBC::insertionAdjRunBFS(cuStinger& custing, adjInsertData* data_h,
	adjInsertData* data_d, vertexId_t treeIdx)
{
	// frontier sizes of bfs queue
	vertexId_t* frontierOffsets = data_h->frontierOffsets;
	// for (int i = 0; i < custing.nv; i++) {
	// 	frontierOffsets[i] = 0;
	// }
	
	bcTree *tree = forest->trees_h[treeIdx];
	tree->currLevel = data_h->ulow_depth;
	
	// Frontier at ulow depth is 1 --> for bfs queue only
	frontierOffsets[tree->currLevel] = 1;
	length_t prevEnd = 1;

	SyncDeviceWithHost(treeIdx);
	while(data_h->bfsQueue.getActiveQueueSize() > 0) {
		printf("While bfs\t\t\tFRONTIER SIZE: %d\n", data_h->bfsQueue.getActiveQueueSize());fflush(NULL);

		allVinA_TraverseEdges_LB<bcOperator::insertionAdjExpandFrontier>(custing,
			(void*) data_d, *cusLB, data_h->bfsQueue);

		data_h->bfsQueue.SyncHostWithDevice();
		copyArrayDeviceToHost(data_d, data_h, 1, sizeof(adjInsertData));

		data_h->bfsQueue.setQueueCurr(prevEnd);
		prevEnd = data_h->bfsQueue.getQueueEnd();

		frontierOffsets[tree->currLevel] = data_h->levelQueue.getActiveQueueSize();

		tree->currLevel++;
		SyncDeviceWithHost(treeIdx);
	}

	// keep currLevel at the lowest frontier depth
	tree->currLevel--;
}


void StreamingBC::insertionAdjRunDA(cuStinger& custing, adjInsertData* data_h,
	adjInsertData* data_d, vertexId_t treeIdx)
{
	bcTree *tree = forest->trees_h[treeIdx];
	
	// Iterate backwards through depths, starting from deepest frontier	
	SyncDeviceWithHost(treeIdx);
	
	vertexId_t* offsets = data_h->frontierOffsets;

	// Set deepest frontier as active queue
	length_t start; // = frontierOffsets[tree->currLevel - 1];
	length_t end; // = frontierOffsets[tree->currLevel];

	length_t levelstart = 0;
	length_t levelend = 0;

	while (tree->currLevel > 0) {
		// printf("Level: %d\tidx: %d\tsize: %d\n", tree->currLevel - level, level, offsets[level]);fflush(NULL);

		
		// if (levelstart != 0 && levelend != 0) {
		// 	// If this is the first pass, don't increment by 1. This helps with
		// 	// level queue boundaries shifting.
		// 	levelstart = levelend;
		// 	levelend = levelstart + data_h->levelQueue.getActiveQueueSize();
		// }

		// First, traverse in levelQueue at level
		levelstart = levelend;
		levelend = levelstart + data_h->levelQueue.getActiveQueueSize();

		data_h->levelQueue.setQueueCurr(levelstart);
		data_h->levelQueue.setQueueEnd(levelend);
		data_h->levelQueue.SyncDeviceWithHost();
		SyncDeviceWithHost(treeIdx);

		allVinA_TraverseEdges_LB<bcOperator::insertionAdjDA>(custing,
			(void*) data_d, *cusLB, data_h->levelQueue);


		// Now, traverse in BFS queue if level is above ulow_depth
		if (tree->currLevel >= data_h->ulow_depth) {

			if (tree->currLevel > data_h->ulow_depth) {
				start = frontierOffsets[tree->currLevel - 1];
				end = frontierOffsets[tree->currLevel];
			} else {
				// on the BFS queue, this frontier only ever holds ulow and it at start of queue
				start = 0;
				end = 1;
			}

			data_h->bfsQueue.setQueueCurr(start);
			data_h->bfsQueue.setQueueEnd(end);
			data_h->bfsQueue.SyncDeviceWithHost();
			SyncDeviceWithHost(treeIdx);

			allVinA_TraverseEdges_LB<bcOperator::insertionAdjDA>(custing,
				(void*) data_d, *cusLB, data_h->bfsQueue);

		}

		tree->currLevel--;
	}

	// while (level < tree->currLevel && offsets[level] > 0) {
	// 	printf("Level: %d\tidx: %d\tsize: %d\n", tree->currLevel - level, level, offsets[level]);fflush(NULL);
	// 	start = end;
	// 	end += offsets[level];

	// 	printf("level size: %d\n", end - start); fflush(NULL);

	// 	data_h->levelQueue.setQueueCurr(start);
	// 	data_h->levelQueue.setQueueEnd(end);
	// 	data_h->levelQueue.SyncDeviceWithHost();
	// 	SyncDeviceWithHost(treeIdx);

	// 	allVinA_TraverseEdges_LB<bcOperator::insertionAdjDA>(custing,
	// 		(void*) data_d, *cusLB, data_h->levelQueue);

	// 	level++;
	// }

	// Now, do it with ulow in the level queue
	// start = end;
	// end += 1;
	// data_h->levelQueue.setQueueCurr(start);
	// // data_h->levelQueue.setQueueEnd(end);

	// // Add ulow to levelqueue
	// data_h->levelQueue.enqueueFromHost(forest->trees_h[treeIdx]->ulow);

	// data_h->levelQueue.SyncDeviceWithHost();
	// SyncDeviceWithHost(treeIdx);

	// allVinA_TraverseEdges_LB<bcOperator::insertionAdjDA>(custing,
	// 	(void*) data_d, *cusLB, data_h->levelQueue);

	
	// while (tree->currLevel > 0) {
	// 	length_t start = frontierOffsets[tree->currLevel];
	// 	length_t end = frontierOffsets[tree->currLevel + 1];

	// 	data_h->bfsQueue.setQueueCurr(start);
	// 	data_h->bfsQueue.setQueueEnd(end);
	// 	data_h->bfsQueue.SyncDeviceWithHost();
	// 	SyncDeviceWithHost(treeIdx);

	// 	allVinA_TraverseEdges_LB<bcOperator::insertionAdjDA>(custing,
	// 		(void*) data_d, *cusLB, data_h->bfsQueue);
	// }

	printf("Copy bc vals onto host\n");
	// Copy over bc values onto host
	copyArrayDeviceToHost(data_h->bcVals, bc, custing.nv, sizeof(float));
	
	// Finally, update all sigma and delta values
	allVinG_TraverseVertices<bcOperator::insertionAdjEnd>(custing, 
		(void*) data_d);
}


void insertionNonadj(cuStinger& custing, vertexId_t* nonadjRoots,
	vertexId_t nonadjCount, bcTree** trees_d)
{

}

void compareStreamVsStatic(cuStinger& custing, StreamingBC& stream, StreamingBC& staticBC)
{
	compdata data_h;
	compdata* data_d = (compdata*) allocDeviceArray(1, sizeof(compdata));

	vertexId_t* dummy_node = (vertexId_t*) allocDeviceArray(1, sizeof(vertexId_t));

	for (int i = 0; i < stream.nr; i++) {
		printf("~~~~~~~~~~~~~~~~~~~Tree idx: %d~~~~~~~~~~~~~~~~~~~\n", i);
		data_h.delta1 = stream.forest->trees_h[i]->delta;
		data_h.sigma1 = stream.forest->trees_h[i]->sigma;

		data_h.delta2 = staticBC.forest->trees_h[i]->delta;
		data_h.sigma2 = staticBC.forest->trees_h[i]->sigma;

		copyArrayHostToDevice(&data_h, data_d, 1, sizeof(compdata));
		allVinA_TraverseVertices<bcOperator::comparison>(custing, (void*) data_d, dummy_node, 1);
	}

	freeDeviceArray(data_d);
	freeDeviceArray(dummy_node);
}

} // cuStingerAlgs namespace
