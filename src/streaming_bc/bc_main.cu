#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>


#include "utils.hpp"
#include "update.hpp"
#include "cuStinger.hpp"

#include <getopt.h>

#include "algs.cuh"
#include "streaming_bc/bc.cuh"
#include "streaming_bc/bc_tree.cuh"

using namespace cuStingerAlgs;
using namespace std;

#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)

typedef struct
{
	bool streaming;
	bool approx;
	
	// number of vertices used. If approx, set here via CLI.
	int numRoots;
	bool verbose;  // print debug info
	int edgesToAdd;  // edges to add
	char *infile;
} program_options;


program_options options;

void printUsageInfo(char **argv)
{
	cout << "Usage: " << argv[0];
	cout << " -i <graph input file> [optional arguments]";
	cout << endl << endl;

	cout << "Options: " << endl;

	cout << "-v                      \tVerbose. Prints debug output";
	cout << " to stdout" << endl;

	cout << "-k <# of src nodes>     \tApproximate BC using a given";
	cout << " number of random source nodes" << endl;

	cout << "-t <# of nodes to add>  \tStreaming BC" << endl;
	cout << endl;
}


void parse_arguments(int argc, char **argv)
{
	int c;
	static struct option long_options[] =
	{
		{"help", no_argument, 0, 'h'},
		{"infile", required_argument, 0, 'i'},
		{"source_nodes", required_argument, 0, 'k'},
		{"stream", required_argument, 0, 't'},  // arg is # of edges to insert
		{"verbose", no_argument, 0,'v'},
		{0,0,0,0} // Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc, argv, "c:de::fg:hi:k:mn::opst:v",
		long_options, &option_index)) != -1)
	{
		switch(c)
		{
			case 'i':
				options.infile = optarg;
			break;

			case 'k':
				options.numRoots = atoi(optarg);
				options.approx = true;
			break;

			case 't':
				options.edgesToAdd = atoi(optarg);
				options.streaming = true;
			break;

			case 'v':
				options.verbose = true;
			break;

			case 'h':
				printUsageInfo(argv);
				exit(0);
			break;

			default: //Fatal error
				cerr << "Internal error parsing arguments." << endl;
				printUsageInfo(argv);
				exit(-1);
		}
	}

	//Handle required command line options here
	if(options.infile == NULL)
	{
		cerr << "Command line error: Graph input file is required.";
		cerr << " Use the -i switch." << endl;
		printUsageInfo(argv);
		exit(-1);
	}
	if(options.approx && (options.numRoots == -1 || options.numRoots < 1))
	{
		cerr << "Command line error: Approximation requested but no";
		cerr << " number of source nodes given. Defaulting to 128.";
		cerr << endl;
		options.numRoots = 128;
	}
	if(options.streaming && (options.edgesToAdd == -1))
	{
		cerr << "Command line error: Streaming requested but no";
		cerr << " number of insertions given. Defaulting to 5.";
		cerr << endl;
		options.edgesToAdd = 5;
	}
}


void generateEdgeUpdates(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst)
{
	cout << "Edge Updates: " << endl;
	for(int32_t e=0; e<numEdges; e++) {
		edgeSrc[e] = rand()%nv;
		edgeDst[e] = rand()%nv;

		cout << "Edge: (" << edgeSrc[e] << ", " << edgeDst[e] << ")";
		cout << endl;
	}
}


void rmat_edge (int64_t * iout, int64_t * jout, int SCALE, double A, double B, double C, double D)
{
  int64_t i = 0, j = 0;
  int64_t bit = ((int64_t) 1) << (SCALE - 1);

  while (1) {
    const double r =  ((double) rand() / (RAND_MAX));//dxor128(env);
    if (r > A) {                /* outside quadrant 1 */
      if (r <= A + B)           /* in quadrant 2 */
        j |= bit;
      else if (r <= A + B + C)  /* in quadrant 3 */
        i |= bit;
      else {                    /* in quadrant 4 */
        j |= bit;
        i |= bit;
      }
    }
    if (1 == bit)
      break;

    /*
      Assuming R is in (0, 1), 0.95 + 0.1 * R is in (0.95, 1.05).
      So the new probabilities are *not* the old +/- 10% but
      instead the old +/- 5%.
    */
    A *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    B *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    C *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    D *= (9.5 + ((double) rand() / (RAND_MAX))) / 10;
    /* Used 5 random numbers. */

    {
      const double norm = 1.0 / (A + B + C + D);
      A *= norm;
      B *= norm;
      C *= norm;
    }
    /* So long as +/- are monotonic, ensure a+b+c+d <= 1.0 */
    D = 1.0 - (A + B + C);

    bit >>= 1;
  }
  /* Iterates SCALE times. */
  *iout = i;
  *jout = j;
}


void generateEdgeUpdatesRMAT(length_t nv, length_t numEdges, vertexId_t* edgeSrc, vertexId_t* edgeDst,double A, double B, double C, double D)
{
	int64_t src,dst;
	int scale = (int)log2(double(nv));
	for(int32_t e=0; e<numEdges; e++){
		rmat_edge(&src,&dst,scale, A,B,C,D);
		edgeSrc[e] = src;
		edgeDst[e] = dst;
	}
}

int main(const int argc, char **argv)
{
	parse_arguments(argc, argv);

	int device = 0;
    cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	bool isDimacs = false;
	bool isSNAP = false;
	bool isRmat = false;
	length_t nv, ne, *off;
	vertexId_t *adj;

	string filename(options.infile);

	isDimacs = filename.find(".graph")==string::npos?false:true;
	isSNAP   = filename.find(".txt")==string::npos?false:true;
	isRmat 	 = filename.find("kron")==string::npos?false:true;

	if(isDimacs){
	    readGraphDIMACS(options.infile, &off, &adj, &nv, &ne, isRmat);
	} else if(isSNAP){
	    readGraphSNAP(options.infile, &off, &adj, &nv, &ne);
	} else {
		cout << "Unknown graph type" << endl;
		exit(0);
	}

	// if not in approx mode, set numRoots to number of vertices
	if (!options.approx) {
		options.numRoots = nv;
	}

	cuStinger custing(defaultInitAllocater,defaultUpdateAllocater);

	cuStingerInitConfig cuInit;
	cuInit.initState = eInitStateCSR;
	cuInit.maxNV = nv + 1;
	cuInit.useVWeight = false;
	cuInit.isSemantic = false;  // Use edge types and vertex types
	cuInit.useEWeight = false;
	
	// CSR data
	cuInit.csrNV  = nv;
	cuInit.csrNE = ne;
	cuInit.csrOff  = off;
	cuInit.csrAdj  = adj;
	cuInit.csrVW  = NULL;
	cuInit.csrEW = NULL;

	custing.initializeCuStinger(cuInit);

	// Store betweenness centrality values here
	float *bc = new float[nv];
	for (int k = 0; k < nv; k++)
	{
		bc[k] = 0;
	}

	vertexId_t root = 0;
	int rootsVisited = 0;

	StreamingBC sbc(options.numRoots);
	sbc.Init(custing);
	sbc.setInputParameters(bc);

	cudaEvent_t ce_start,ce_stop;
	start_clock(ce_start, ce_stop);

	sbc.Run(custing);

	float totalTime = end_clock(ce_start, ce_stop);
	cout << "Total time for Betweenness Centrality Computation: " << totalTime << endl;

	// Now, insert a random edge
	vertexId_t src = rand() % nv;
	vertexId_t dst = rand() % nv;
	
	cout << "About to insert edge: (" << src << ", " << dst << ")" << endl;
	start_clock(ce_start, ce_stop);
	sbc.InsertEdge(custing, src, dst);

	totalTime = end_clock(ce_start, ce_stop);
	cout << "Done inserting. Total time taken:  " << totalTime  << endl;

	if (options.verbose) {
		cout << "RESULTS: " << endl;

		for (int k = 0; k < nv; k++) {
			cout << "[ " << k  << " ]: " << bc[k] << endl;
		}
	}

	cout << "=======================================" << endl;
	cout << "Now doing brute force edge insertion" << endl;


	// Add that same edge into the graph and run static bc on it
	// TODO: figure out how to add edge
	length_t allocs;
	// auto bud = new BatchUpdateData(1, true, custing.nv);
	BatchUpdateData bud(1 , true, custing.nv);
	vertexId_t *srcs = bud.getSrc();
	vertexId_t *dsts = bud.getDst();
	srcs[0] = src;
	dsts[0] = dst;

	BatchUpdate bu = BatchUpdate(bud);

	custing.edgeInsertions(bu, allocs);

	float *bc_static = new float[nv];
	for (int k = 0; k < nv; k++) {
		bc_static[k] = 0;
	}	

	StreamingBC sbc2(options.numRoots);
	sbc2.Init(custing);
	sbc2.setInputParameters(bc_static);

	start_clock(ce_start, ce_stop);

	sbc2.Run(custing);

	totalTime = end_clock(ce_start, ce_stop);
	cout << "Done with static. Total time taken:  " << totalTime  << endl;

	bool same = true;
	for (int k = 0; k < nv; k++) {
		if (bc_static[k] != bc[k]) {
			same = false;
			break;
		}
	}

	cout << "Are they same?   :: " << (same?"true":"false") << endl;

	// free resources
	sbc.Reset();
	sbc.Release();

	// free resources
	sbc2.Reset();
	sbc2.Release();

	// Free memory
	custing.freecuStinger();

	free(off);
	free(adj);

	delete[] bc;
	delete[] bc_static;

    return 0;
}
