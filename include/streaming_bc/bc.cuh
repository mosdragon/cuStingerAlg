#pragma once

#include "algs.cuh"
#include "load_balance.cuh"
#include "streaming_bc/bc_tree.cuh"

// Betweenness Centrality

#define UNTOUCHED 0
#define UP 128
#define DOWN 64

namespace cuStingerAlgs {

typedef struct {
	// diffs_d is an array of size numRoots that shows stores d[src] - d[dst] in each position
	vertexId_t* diffs_d;
	// a device array of tree pointers
	bcTree** trees_d;

	length_t numRoots;
	vertexId_t src;  // the src of the edge being inserted
	vertexId_t dst;  // the dst of the edge being inserted
} depthDiffs;


typedef struct {
	vertexQueue levelQueue;
	vertexQueue bfsQueue;
	length_t numRoots;
	vertexId_t treeIdx;
	bcTree** trees_d;

	char* t;  // will take on values of UNTOUCHED, DOWN, or UP
	vertexId_t* dP;
	vertexId_t* sigmaHat;
	float* deltaHat;
	float* bcVals;

	// Host-only tracking of levelQueue offsets
	vertexId_t* frontierOffsets;

} adjInsertData;

typedef struct {
	float* delta1;
	float* delta2;
	vertexId_t* sigma1;
	vertexId_t* sigma2;
} compdata;



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

// TODO: Make private again
// private:
	bcForest *forest;
	bcTree **trees_d;  // a device copy of the host array forest->trees_d
	float *bc;  // the actual bc values array on the host

	// a float array which will contain a copy of the device delta array
	// during dependency accumulation
	float *host_deltas;
	cusLoadBalance* cusLB;  // load balancer
	length_t nr;  // number of roots/trees
	bool approx;


	// private helper functions
	void insertionAdj(cuStinger& custing, vertexId_t* adjRoots_h,
		vertexId_t* adjRoots_d, vertexId_t adjCount);

	void insertionAdjRunBFS(cuStinger& custing, adjInsertData* data_h,
		adjInsertData* data_d, vertexId_t treeIdx);

	void insertionAdjRunDA(cuStinger& custing, adjInsertData* data_h,
		adjInsertData* data_d, vertexId_t treeIdx);

};


class bcOperator {
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


	// Computes the depth differences between src and dst and
	// assigns vertices ulow and uhigh
	static __device__ __forceinline__ void preprocessEdge(cuStinger* custing,
		vertexId_t treeIdx, void* metadata)
	{
		depthDiffs* dDiffs_d = (depthDiffs*) metadata;

		vertexId_t src = dDiffs_d->src;
		vertexId_t dst = dDiffs_d->dst;

		vertexId_t* depths = dDiffs_d->trees_d[treeIdx]->d;
		vertexId_t diff = depths[dst] - depths[src];

		dDiffs_d->diffs_d[treeIdx] = diff;

		// assign ulow and uhigh
		bcTree *tree_d = dDiffs_d->trees_d[treeIdx];
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


	// Adjacent Insertion: Initialization
	static __device__ __forceinline__ void insertionAdjInit(cuStinger* custing,
		vertexId_t treeIdx, void* metadata)
	{
		adjInsertData* data_d = (adjInsertData*) metadata;
		bcTree* tree = data_d->trees_d[treeIdx];

		data_d->treeIdx = treeIdx;

		for (vertexId_t i = 0; i < custing->nv; i++) {
			data_d->dP[i] = 0;
			data_d->t[i] = UNTOUCHED;
			data_d->sigmaHat[i] = tree->sigma[i];
			data_d->deltaHat[i] = 0;
 		}

 		vertexId_t ulow = tree->ulow;
		vertexId_t uhigh = tree->uhigh;

		data_d->levelQueue.enqueue(ulow);
		data_d->bfsQueue.enqueue(ulow);

		data_d->t[ulow] = DOWN;
		data_d->dP[ulow] = tree->sigma[uhigh];
		data_d->sigmaHat[ulow] += data_d->dP[ulow];
	}

	// Adjacent Insertion: BFS Expand Frontier
	static __device__ __forceinline__ void insertionAdjExpandFrontier(
		cuStinger* custing, vertexId_t src, vertexId_t dst, void* metadata)
	{
		vertexId_t v = src;
		vertexId_t w = dst;

		adjInsertData* data_d = (adjInsertData*) metadata;
		bcTree *tree = data_d->trees_d[data_d->treeIdx];

		if (tree->d[w] == tree->d[v] + 1) {
			if (data_d->t[w] == UNTOUCHED) {
				data_d->bfsQueue.enqueue(w);
				data_d->levelQueue.enqueue(w);
				data_d->t[w] = DOWN;
				tree->d[w] = tree->d[v] + 1;
				data_d->dP[w] = data_d->dP[v];

			} else {
				// use atomics
				// data_d->dP[w] += data_d->dP[v];
				atomicAdd(data_d->dP + w, data_d->dP[v]);
			}

			data_d->sigmaHat[w] += data_d->dP[v];
		}
	}


	// Adjacent Insertion: Dependency accumulation
	static __device__ __forceinline__ void insertionAdjDA(cuStinger* custing,
		vertexId_t src, vertexId_t dst, void* metadata)
	{
		vertexId_t v = src;
		vertexId_t w = dst;

		adjInsertData* data_d = (adjInsertData*) metadata;
		bcTree *tree = data_d->trees_d[data_d->treeIdx];

		vertexId_t* sigmaHat = data_d->sigmaHat;
		float* deltaHat = data_d->deltaHat;

		// ensure v belongs to P[w]
		if (tree->d[w] == tree->d[v] + 1) {
			if (data_d->t[v] == UNTOUCHED) {
				// TODO: figure out how to enqueue to Q[level - 1]
				// Why does this never happen??
				printf("Enqueue to level: Q[level - 1]\n");
				data_d->levelQueue.enqueue(v);
				data_d->t[v] = UP;
				deltaHat[v] = tree->delta[v];
			}

			// use atomics
			// deltaHat[v] += ((float) sigmaHat[v] / sigmaHat[w]) * (1 + deltaHat[w]);
			atomicAdd(deltaHat + v, ((float) sigmaHat[v] / sigmaHat[w]) * (1 + deltaHat[w]));


			if (data_d->t[v] == UP && (v != tree->uhigh || w != tree->ulow)) {
				// use atomics
				// deltaHat[v] -= ((float) tree->sigma[v] / tree->sigma[w]) * (1 + tree->delta[w]);
				atomicAdd(deltaHat + v, -1 * ((float) tree->sigma[v] / tree->sigma[w]) * (1 + tree->delta[w]));
			}

			if (w != tree->root) {
				// update BC values
				// use atomics
				// data_d->bcVals[w] += deltaHat[w] - tree->delta[w];
				atomicAdd(data_d->bcVals + w, deltaHat[w] - tree->delta[w]);

			}
		}
	}

	// Adjacent Insertion: Completion / Dependency Accumulation part 2
	static __device__ __forceinline__ void insertionAdjEnd(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		// Updates sigma and delta values
		adjInsertData* data_d = (adjInsertData*) metadata;
		bcTree *tree = data_d->trees_d[data_d->treeIdx];

		tree->sigma[src] = data_d->sigmaHat[src];
		if (data_d->t[src] == UNTOUCHED) {
			tree->delta[src] = data_d->deltaHat[src];
		}
	}


	static __device__ __forceinline__ void comparison(cuStinger* custing,
		vertexId_t src, void* metadata)
	{
		compdata* data = (compdata*) metadata;

		printf("=========SIGMAS=======\n");
		for (int i = 0; i < custing->nv; i++) {
			if (data->sigma1[i] != data->sigma2[i]) {
				printf("Mismatch:\tidx: %d [%d] != [%d]\n", i, data->sigma1[i], data->sigma2[i]);
			}
		}

		printf("=========DELTAS=======\n");
		for (int i = 0; i < custing->nv; i++) {
			if (data->delta1[i] != data->delta2[i]) {
				printf("Mismatch:\tidx: %d [%f] != [%f]\n", i, data->delta1[i], data->delta2[i]);
			}
		}

	}

}; // bcOperator


/* FOR DEBUG ONLY */

void compareStreamVsStatic(cuStinger& custing, StreamingBC& stream, StreamingBC& staticBC);

/* END DEBUG */


// diffs_h is an array of size numRoots that shows stores d[src] - d[dst] in each position
void getDepthDifferences(cuStinger& custing, vertexId_t src, vertexId_t dst,
	length_t numRoots, vertexId_t* diffs_h, bcTree** trees_d);


// Takes a diff_h array and creates an array with consecutive sections
vertexId_t* buildCaseArray(vertexId_t* diffs_h, length_t numRoots,
	vertexId_t& size, vertexId_t& adj, vertexId_t& nonadj);


// void insertionAdj(cuStinger& custing, vertexId_t* adjRoots_d,
// 	vertexId_t adjCount, bcForest *forest, bcTree** trees_d);

void insertionNonadj(cuStinger& custing, vertexId_t* nonadjRoots,
	vertexId_t nonadjCount, bcTree** trees_d);

} //Namespace