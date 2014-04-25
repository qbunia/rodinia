#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define TRANSFER_GRAPH_NODE 1

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    char *input_f;

    int* h_cost;
    int* h_graph_edges;
	
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
    
	input_f = argv[1];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
   
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;
	
	fscanf(fp,"%d",&edge_list_size);
	
	int id,cost;
	h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp); 

#pragma acc data create(h_updating_graph_mask[0:no_of_nodes]) \
	create(h_graph_mask[0:no_of_nodes],h_graph_visited[0:no_of_nodes]) \
	create(h_graph_nodes[0:no_of_nodes], h_graph_edges[0:edge_list_size]) \
	copyout(h_cost[0:no_of_nodes])
{
	#pragma acc update device(h_graph_nodes[0:no_of_nodes]) async(TRANSFER_GRAPH_NODE)

	#pragma acc parallel loop
	for( unsigned int i = 0; i < no_of_nodes; i++)
	{
		h_updating_graph_mask[i]=false;
		h_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}
	
	#pragma acc kernels present(h_graph_mask[0:no_of_nodes],h_graph_visited[0:no_of_nodes])
	{
	    //set the source node as true in the mask
	    h_graph_mask[source]=true;
		h_graph_visited[source]=true;
	}

	// allocate mem for the result on host side
	h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	#pragma acc parallel loop
	for(int i=0;i<no_of_nodes;i++) {
		h_cost[i]=-1;
		if(i == source) h_cost[source]=0;
	}
	
	// finish transfer node and edge to target
	#pragma acc update device(h_graph_edges[0:edge_list_size])
	#pragma acc wait(TRANSFER_GRAPH_NODE)

	printf("Start traversing the tree\n");

	int k=0;
    
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

		#pragma acc parallel loop
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
			h_graph_mask[tid]=false;
			for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
				{
				int id = h_graph_edges[i];
				if(!h_graph_visited[id])
					{
					h_cost[id]=h_cost[tid]+1;
					h_updating_graph_mask[id]=true;
					}
				}
			}
		}

		#pragma acc parallel loop vector reduction(||:stop)
  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);

} /* end acc data */

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

}

