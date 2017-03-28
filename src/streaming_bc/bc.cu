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

	for (length_t k = 0; k < nr; k++)
	{
		forest->trees_h[k]->queue.Init(custing.nv);
	}

	host_deltas = new float[custing.nv];
	cusLB = new cusLoadBalance(custing.nv);

	// Keep a device copy of the array forest->trees_d
	trees_d = (bcTree**) allocDeviceArray(nr, sizeof(bcTree*));
	copyArrayHostToDevice(forest->trees_d, trees_d, nr, sizeof(bcTree*));

	if (nr == custing.nv)
	{
		approx = false;
	}
	Reset();
}


void StreamingBC::Reset()
{
	for (length_t k = 0; k < nr; k++)
	{
		bcTree *hostBcTree = forest->trees_h[k];
		hostBcTree->queue.resetQueue();
		hostBcTree->currLevel = 0;

		if (approx) {
			hostBcTree->root = rand() % forest->nv;
		} else {
			hostBcTree->root = k;
		}

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

	while (getLevel(k) >= 0)
	{
		length_t start = hostBcTree->offsets[getLevel(k)];
		length_t end = hostBcTree->offsets[getLevel(k) + 1];

		// // set queue start and end so the queue holds all nodes in one frontier
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

	// // Finally, update the bc values
	for (vertexId_t w = 0; w < hostBcTree->nv; w++)
	{
		if (w != hostBcTree->root)
		{
			bc[w] += host_deltas[w];
		}
	}
}


void StreamingBC::InsertEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
{
	vertexId_t *diffs_h = new vertexId_t[nr];
	getDepthDifferences(custing, src, dst, nr, diffs_h, trees_d);

	SyncHostWithDevice();  // copy the updated ulow, uhigh assignments to host

	// the count of adjacent cases and nonadjacent cases
	vertexId_t adj = 0;
	vertexId_t nonadj = 0;
	vertexId_t size;  // the overall size of the case array

	vertexId_t* caseArray_h = buildCaseArray(diffs_h, nr, size, adj, nonadj);

	vertexId_t* caseArray_d = (vertexId_t*) allocDeviceArray(size, sizeof(vertexId_t));
	copyArrayHostToDevice(caseArray_h, caseArray_d, size, sizeof(vertexId_t));


	insertionAdj(custing, caseArray_d, adj);

	// insertionNonadj(custing, caseArray_h + adj, nonadj, trees_d);

	freeDeviceArray(caseArray_d);
	delete[] diffs_h;
	delete[] caseArray_h;
}


void StreamingBC::RemoveEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
{

}


void getDepthDifferences(cuStinger& custing, vertexId_t src, vertexId_t dst,
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

	// now, use a streaming operator to get the depth differences and ulow and uhigh assignments
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
}


vertexId_t* buildCaseArray(vertexId_t* diffs_h, length_t numRoots,
	vertexId_t& size, vertexId_t& adj, vertexId_t& nonadj)
{
	for (int k = 0; k < numRoots; k++) {
		if (abs(diffs_h[k]) == 1) {
			adj++;
		} else if (abs(diffs_h[k]) > 1) {
			nonadj++;
		}
	}

	// positions in the array of where to place each case type
	vertexId_t posAdj = 0;  // index used for adjacent cases
	vertexId_t posNonadj = adj;  // index used for nonadjacent cases
	size = adj + nonadj;

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

void StreamingBC::insertionAdj(cuStinger& custing, vertexId_t* adjRoots_d,
	vertexId_t adjCount)
{

	printf("insertionAdj begin\n");
	adjInsertData data_h;
	data_h.trees_d = trees_d;
	data_h.numRoots = forest->numRoots;
	data_h.upQueue.Init(custing.nv + 1);
	data_h.downQueue.Init(custing.nv + 1);

	// TODO: optimize these allocations
	data_h.t =  (char*) allocDeviceArray(custing.nv, sizeof(char));
	data_h.dP = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));
	data_h.sigmaHat = (vertexId_t*) allocDeviceArray(custing.nv, sizeof(vertexId_t));

	adjInsertData* data_d = (adjInsertData*) allocDeviceArray(1,
		sizeof(adjInsertData));
	copyArrayHostToDevice(&data_h, data_d, 1, sizeof(adjInsertData));

	// run algorithm sequentially for each adjacent-case tree
	for (int k = 0; k < adjCount; k++) {
		printf("Loop begin\n");
		// 1: INITIALIZATION
		// Clear upQueue and downQueue
		data_h.upQueue.resetQueue();
		data_h.downQueue.resetQueue();

		allVinA_TraverseVertices<bcOperator::initAdjInsert>(custing,
			(void*) data_d, adjRoots_d + k, 1);

		insertionAdjRunBFS(custing, &data_h, data_d, k);

		printf("Finished initialization\n");

		// run the operator on a single tree
		// allVinA_TraverseEdges_LB<>();
	}

	freeDeviceArray(data_d);
	freeDeviceArray(data_h.t);
	freeDeviceArray(data_h.dP);
	freeDeviceArray(data_h.sigmaHat);
	printf("Finished adj cases\n");
}

void StreamingBC::insertionAdjRunBFS(cuStinger& custing, adjInsertData* data_h,
	adjInsertData* data_d, vertexId_t treeIdx)
{
	length_t prevEnd = 1;
	bcTree *tree = forest->trees_h[treeIdx];
	tree->currLevel = 0;
	SyncDeviceWithHost(treeIdx);
	while(data_h->downQueue.getActiveQueueSize() > 0)
	{
		allVinA_TraverseEdges_LB<bcOperator::insertionAdjExpandFrontier>(custing,
			(void*) data_d, *cusLB, data_h->downQueue);
		SyncHostWithDevice(treeIdx);

		data_h->downQueue.setQueueCurr(prevEnd);
		prevEnd = data_h->downQueue.getQueueEnd();

		tree->currLevel++;
		SyncDeviceWithHost(treeIdx);
	}
}


void insertionNonadj(cuStinger& custing, vertexId_t* nonadjRoots,
	vertexId_t nonadjCount, bcTree** trees_d)
{

}

} // cuStingerAlgs namespace
