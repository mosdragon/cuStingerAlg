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
	
	while( hostBcTree->queue.getActiveQueueSize() > 0)
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
	getDepthDifferences(src, dst, nr, diffs_h, trees_d);

	int same = 0;
	
	int adj = 0;
	int nonadj = 0;

	int adjRev = 0;
	int nonadjRev = 0;
	for (int k = 0; k < nr; k++)
	{
		if (diffs_h[k] == 0) {
			same++;
		
		} else if (diffs_h[k] == 1) {
			// d[src] - d[dst] == 1
			adj++;

		} else if (diffs_h[k] == -1) {
			// d[dst] - d[src] == 1
			adjRev++;
		
		} else if (diffs_h[k] > 1) {
			// d[src] - d[dst] > 1
			nonadj++;
		} else {
			// d[dst] - d[src] > 1
			nonadjRev++;
		}
	}

	printf("Same level cases: %d\n", same);
	printf("Adjacent cases: %d\n", adj);
	printf("Non-adjacent cases: %d\n", nonadj);

	for (int k = 0; k < nr; k++) {
		if (diffs_h[k] == 1) {

		} else if (diffs_h[k] == -1) {

		}
	}

	delete[] diffs_h;
}


void StreamingBC::RemoveEdge(cuStinger& custing, vertexId_t src, vertexId_t dst)
{

}


// diffs_h is an array of size K that shows stores d[src] - d[dst] in each position
void getDepthDifferences(vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_h, bcTree** trees_d)
{
	// TODO: Load balance this operation later
	vertexId_t* diffs_d = (vertexId_t*) allocDeviceArray(numRoots, sizeof(vertexId_t));

	dim3 numBlocks(1, 1);
	int32_t threads = 32;
	dim3 threadsPerBlock(threads, 1);
	int32_t treesPerThreadBlock = 128;

	numBlocks.x = ceil((float) numRoots / (float) treesPerThreadBlock);

	getDepthDifferences_device<<<numBlocks, threadsPerBlock>>>(src, dst,
		numRoots, diffs_d, trees_d, treesPerThreadBlock);

	// copy the differences back over to host array
	copyArrayDeviceToHost(diffs_d, diffs_h, numRoots, sizeof(vertexId_t));
	freeDeviceArray(diffs_d);
}


// diffs_d is just like diffs_h except that it points to GPU memory
__global__ void getDepthDifferences_device(vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_d, bcTree** trees_d,
	int32_t treesPerThreadBlock)
{
	vertexId_t treeIdx_init = blockIdx.x * treesPerThreadBlock;
	length_t K = numRoots;

	for (vertexId_t offset = 0; offset < treesPerThreadBlock; offset++) {
		vertexId_t treeIdx = treeIdx_init + offset;
		
		if(treeIdx >= K) {
			return;
		}
		// Set diff_d[treeIdx] to the different in depths between src and dst
		// at the treeIdx
		diffs_d[treeIdx] = trees_d[treeIdx]->d[src] - trees_d[treeIdx]->d[dst];
	}

}


// uhigh is the vertex that has lesser depth and ulow has greater depth
void insertAdjacentLevel(bcTree** trees_d, length_t numRoots, 
	vertexId_t uhigh, vertexId_t ulow)
{

}

__global__ void insertAdjacentLevel_device(bcTree** trees_d, length_t numRoots,
	vertexId_t uhigh, vertexId_t ulow, int32_t treesPerThreadBlock)
{
	vertexId_t treeIdx_init = blockIdx.x * treesPerThreadBlock;
	length_t K = numRoots;

	for (vertexId_t offset = 0; offset < treesPerThreadBlock; offset++) {
		vertexId_t treeIdx = treeIdx_init + offset;
		
		if(treeIdx >= K) {
			return;
		}
		// Set diff_d[treeIdx] to the different in depths between src and dst
		// at the treeIdx
		// diffs_d[treeIdx] = trees_d[treeIdx]->d[src] - trees_d[treeIdx]->d[dst];
	}
}


} // cuStingerAlgs namespace