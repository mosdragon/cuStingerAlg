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
	void SyncDeviceWithHost(length_t k)
	{
		copyArrayHostToDevice(forest->trees_h[k], forest->trees_d[k], 1, sizeof(bcTree));
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


// diffs_h is an array of size K that shows stores d[src] - d[dst] in each position
void getDepthDifferences(vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_h, bcTree** trees_d);


// diffs_d is just like diffs_h except that it points to GPU memory
__global__ void getDepthDifferences_device(vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_d, bcTree** trees_d,
	int32_t treesPerThreadBlock);


// uhigh is the vertex that has lesser depth and ulow has greater depth
void insertAdjacentLevel(bcTree** trees_d, length_t numRoots, 
	vertexId_t uhigh, vertexId_t ulow);

__global__ void insertAdjacentLevel_device(bcTree** trees_d, length_t numRoots,
	vertexId_t uhigh, vertexId_t ulow, int32_t treesPerThreadBlock);


} //Namespace