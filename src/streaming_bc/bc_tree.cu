#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include "algs.cuh"
#include "streaming_bc/bc_tree.cuh"

namespace cuStingerAlgs {

// user is responsible for calling destroyHostBcTree
bcTree* createHostBcTree(length_t nv)
{
	bcTree *tree_h = new bcTree;
	tree_h->offsets = new float[nv];
	tree_h->nv = nv;
	return tree_h;
}


// user is responsible for calling destroyDeviceBcTree
// must provide number of vertices (nv) and pointer to host bcTree
bcTree* createDeviceBcTree(length_t nv, bcTree *tree_h)
{
	int size = sizeof(bcTree);
	size += 2 * nv * sizeof(vertexId_t);  // d and sigma arrays
	size += nv * sizeof(float);  // delta array

	char *starting_point = (char *) allocDeviceArray(1, size);
	bcTree *tree_d = (bcTree*) starting_point;

	// pointer arithmetic for d, sigma, delta pointers
	// these are actual memory locations on the device for the arrays
	char *d = starting_point + sizeof(bcTree);  // start where tree_d ends
	char *sigma = d + nv * sizeof(vertexId_t);  // start where d ends
	char *delta = sigma + nv * sizeof(vertexId_t);  // start where sigma ends

	tree_h->d = (vertexId_t *) d;
	tree_h->sigma = (vertexId_t *) sigma;
	tree_h->delta = (float *) delta;

	copyArrayHostToDevice(tree_h, tree_d, 1, sizeof(bcTree));

	return tree_d;
}


void destroyDeviceBcTree(bcTree* tree_d)
{
	freeDeviceArray(tree_d);
}


void destroyHostBcTree(bcTree* tree_h)
{
	delete[] tree_h->offsets;
	delete tree_h;
}

// user is responsible for calling destroyBcForest
bcForest* createBcForest(length_t nv, length_t numRoots)
{
	bcForest *forest = new bcForest;
	forest->trees_h = new bcTree*[numRoots];
	forest->trees_d = new bcTree*[numRoots];

	for (vertexId_t k = 0; k < numRoots; k++)
	{
		bcTree *tree_h = createHostBcTree(nv);
		forest->trees_h[k] = tree_h;
		forest->trees_d[k] = createDeviceBcTree(nv, tree_h);
	}
	forest->numRoots = numRoots;
	forest->nv = nv;

	return forest;
}


void destroyBcForest(bcForest *forest, length_t numRoots)
{
	for (vertexId_t k = 0; k < numRoots; k++)
	{
		destroyDeviceBcTree(forest->trees_d[k]);
		destroyHostBcTree(forest->trees_h[k]);
	}
	delete[] forest->trees_d;
	delete[] forest->trees_h;
	delete forest;
}

}  // namespace