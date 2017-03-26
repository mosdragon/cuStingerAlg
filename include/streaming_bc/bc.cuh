#pragma once

#include "algs.cuh"
#include "load_balance.cuh"
#include "streaming_bc/bc_tree.cuh"

// Betweenness Centrality

namespace cuStingerAlgs {

class StreamingBC:public StreamingAlgorithm {
public:	
	
	StreamingBC(length_t numRoots) {
		nr = numRoots;
	}

	void Init(cuStinger& custing);
	void Reset();
	void Run(cuStinger& custing);
	void Release();

	void InsertEdge(cuStinger& custing, vertexId_t src, vertexId_t dst);
	void RemoveEdge(cuStinger& custing, vertexId_t src, vertexId_t dst);

	void SyncHostWithDevice(length_t k)
	{
		copyArrayDeviceToHost(forest->trees_d[k], forest->trees_h[k], 1, sizeof(bcTree));
	}

	void SyncHostWithDevice()
	{
		for (int k = 0; k < nr; k++) {
			SyncHostWithDevice(k);
		}
	}

	void SyncDeviceWithHost(length_t k)
	{
		copyArrayHostToDevice(forest->trees_h[k], forest->trees_d[k], 1, sizeof(bcTree));
	}

	void SyncDeviceWithHost()
	{
		for (int k = 0; k < nr; k++) {
			SyncDeviceWithHost(k);
		}
	}

	void RunBfsTraversal(cuStinger& custing, length_t k);
	void DependencyAccumulation(cuStinger& custing, length_t k);
	
	length_t getLevel(length_t k) { return forest->trees_h[k]->currLevel; }

	// Must pass a pointer to bc values (of length custing.nv)
	void setInputParameters(float *bc_array);

private:
	bcForest *forest;
	bcTree **trees_d;  // a device copy of the host array forest->trees_d
	float *bc;  // the actual bc values array on the host

	// a float array which will contain a copy of the device delta array
	// during dependency accumulation
	float *host_deltas;
	cusLoadBalance* cusLB;  // load balancer
	length_t nr;  // number of roots/trees
	bool approx;
};

typedef struct {
	// diffs_d is an array of size numRoots that shows stores d[src] - d[dst] in each position
	vertexId_t* diffs_d;
	// a device array of tree pointers
	bcTree** trees_d;
	
	length_t numRoots;
	vertexId_t src;  // the src of the edge being inserted
	vertexId_t dst;  // the dst of the edge being inserted
} depthDiffs;


class bcOperator:public StreamingAlgorithm {
public:

	static __device__ __forceinline__ void bcExpandFrontier(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;
		vertexId_t nextLevel = bcd->currLevel + 1;

		vertexId_t v = src;
		vertexId_t w = dst;

		vertexId_t prev = atomicCAS(bcd->d + w, INT32_MAX, nextLevel);
		if (prev == INT32_MAX) {
			bcd->queue.enqueue(w);
		}
		if (bcd->d[w] == nextLevel) {
			atomicAdd(bcd->sigma + w, bcd->sigma[v]);
		}

	}

	// Use macro to clear values in arrays to 0. Set level to INF
	static __device__ __forceinline__ void setupArrays(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;
		bcd->d[src] = INT32_MAX;
		bcd->sigma[src] = 0;
		bcd->delta[src] = 0.0;
	}


	// Computes the depth differences between src and dst and
	// assigns vertices ulow and uhigh
	static __device__ __forceinline__ void preprocessEdge(cuStinger* custing,
		vertexId_t treeRoot, void* metadata)
	{
		depthDiffs* dDiffs_d = (depthDiffs*) metadata;

		vertexId_t src = dDiffs_d->src;
		vertexId_t dst = dDiffs_d->dst;

		vertexId_t* depths = dDiffs_d->trees_d[treeRoot]->d;
		vertexId_t diff = depths[dst] - depths[src];

		dDiffs_d->diffs_d[treeRoot] = diff;

		// assign ulow and uhigh
		bcTree *tree_d = dDiffs_d->trees_d[treeRoot];
		if (diff > 0) {
			// if difference is positive, dst is "below" src
			tree_d->ulow = dst;
			tree_d->uhigh = src;
		} else if (diff < 0) {
			// if difference is negative, src is "below" dst
			tree_d->ulow = src;
			tree_d->uhigh = dst;
		}
	}


	// Dependency accumulation for one frontier
	static __device__ __forceinline__ void dependencyAccumulation(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		bcTree* bcd = (bcTree*) metadata;

		vertexId_t *d = bcd->d;  // depth
		vertexId_t *sigma = bcd->sigma;
		float *delta = bcd->delta;

		vertexId_t v = src;
		vertexId_t w = dst;

		if (d[w] == d[v] + 1)
		{
			atomicAdd(delta + v, ((float) sigma[v] / (float) sigma[w]) * (1 + delta[w]));
		}
	}

}; // bcOperator


// diffs_h is an array of size numRoots that shows stores d[src] - d[dst] in each position
void getDepthDifferences(cuStinger& custing, vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_h, bcTree** trees_d);


// Takes a diff_h array and creates an array with consecutive sections
vertexId_t* buildCaseArray(vertexId_t* diffs_h, length_t numRoots,
	vertexId_t& size, vertexId_t& adj, vertexId_t& nonadj,
	vertexId_t& adjRev, vertexId_t& nonadjRev);

} //Namespace