#ifndef STREAMING_BC_TREE
#define STREAMING_BC_TREE

#include "algs.cuh"

namespace cuStingerAlgs {

typedef struct {
    length_t nv;
    vertexId_t root;
    vertexId_t currLevel;

    vertexId_t* d;  // depth of each vertex
    vertexId_t* sigma;
    float* delta;

    vertexQueue queue;

    // offsets stores ending position of each frontier in the queue.
    // Used during dependency accumulation. Stored in host memory
    float* offsets;
} bcTree;


typedef struct {
    bcTree **trees_h;  // array of pointer locations on host
    bcTree **trees_d;  // array of pointer locations on device
    length_t nv;  // number of vertices
    length_t numRoots;  // number of roots/trees used
} bcForest;


// user is responsible for calling destroyHostBcTree
bcTree* createHostBcTree(length_t nv);

// user is responsible for calling destroyDeviceBcTree
// must provide number of vertices (nv) and pointer to host bcTree
bcTree* createDeviceBcTree(length_t nv, bcTree *tree_h);

void destroyDeviceBcTree(bcTree* tree_d);

void destroyHostBcTree(bcTree* tree_h);

// user is responsible for calling destroyBcForest
bcForest* createBcForest(length_t nv, length_t numRoots);

void destroyBcForest(bcForest *forest, length_t numRoots);

}  // namespace

#endif