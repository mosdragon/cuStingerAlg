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

void StreamingBC::InsertEdges(vertexId_t src, vertexId_t dst)
{

}


void StreamingBC::RemoveEdges(vertexId_t src, vertexId_t dst)
{

}


} // cuStingerAlgs namespace 