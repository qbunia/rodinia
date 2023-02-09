#include "rex_kmp.h" 
char OUT__2__4476__BFSGraph__153__id__ = 0;
struct __tgt_offload_entry OUT__2__4476__BFSGraph__153__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__2__4476__BFSGraph__153__id__)), "OUT__2__4476__BFSGraph__153__kernel__", 0, 0, 0};
char OUT__1__4476__BFSGraph__135__id__ = 0;
struct __tgt_offload_entry OUT__1__4476__BFSGraph__135__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__4476__BFSGraph__135__id__)), "OUT__1__4476__BFSGraph__135__kernel__", 0, 0, 0};
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
FILE *fp;
// Structure to hold a node information

struct Node 
{
  int starting;
  int no_of_edges;
}
;
void BFSGraph(int argc,char **argv);

void Usage(int argc,char **argv)
{
  fprintf(stderr,"Usage: %s <input_file> [<num_teams>] [<num_threads>]\n",argv[0]);
}

double get_time()
{
  struct timeval t;
  gettimeofday(&t,(void *)0);
  return ((double )t . tv_sec) + t . tv_usec * 1e-6;
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////

int main(int argc,char **argv)
{
  int status = 0;
  BFSGraph(argc,argv);
}
////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////

void BFSGraph(int argc,char **argv)
{
  int no_of_nodes = 0;
  int edge_list_size = 0;
  char *input_f;
  int omp_num_teams = 256;
  int omp_num_threads = 1024;
  if (argc < 2) {
    Usage(argc,argv);
    exit(0);
  }
  input_f = argv[1];
  if (argc > 2) {
    omp_num_teams = atoi(argv[2]);
  }
  if (argc > 3) {
    omp_num_threads = atoi(argv[3]);
  }
  printf("Reading File\n");
// Read in Graph from a file
  fp = fopen(input_f,"r");
  if (!fp) {
    printf("Error Reading graph file\n");
    return ;
  }
  int source = 0;
  fscanf(fp,"%d",&no_of_nodes);
// allocate host memory
  struct Node *h_graph_nodes = (struct Node *)(malloc(sizeof(struct Node ) * no_of_nodes));
  _Bool *h_graph_mask = (_Bool *)(malloc(sizeof(_Bool ) * no_of_nodes));
  _Bool *h_updating_graph_mask = (_Bool *)(malloc(sizeof(_Bool ) * no_of_nodes));
  _Bool *h_graph_visited = (_Bool *)(malloc(sizeof(_Bool ) * no_of_nodes));
  int start;
  int edgeno;
// initalize the memory
  for (unsigned int i = 0; i < no_of_nodes; i++) {
    fscanf(fp,"%d %d",&start,&edgeno);
    h_graph_nodes[i] . starting = start;
    h_graph_nodes[i] . no_of_edges = edgeno;
    h_graph_mask[i] = 0;
    h_updating_graph_mask[i] = 0;
    h_graph_visited[i] = 0;
  }
// read the source node from the file
  fscanf(fp,"%d",&source);
// source=0; //tesing code line
// set the source node as true in the mask
  h_graph_mask[source] = 1;
  h_graph_visited[source] = 1;
  fscanf(fp,"%d",&edge_list_size);
  int id;
  int cost;
  int *h_graph_edges = (int *)(malloc(sizeof(int ) * edge_list_size));
  for (int i = 0; i < edge_list_size; i++) {
    fscanf(fp,"%d",&id);
    fscanf(fp,"%d",&cost);
    h_graph_edges[i] = id;
  }
  if (fp) 
    fclose(fp);
// allocate mem for the result on host side
  int *h_cost = (int *)(malloc(sizeof(int ) * no_of_nodes));
  for (int i = 0; i < no_of_nodes; i++) 
    h_cost[i] = - 1;
  h_cost[source] = 0;
  printf("Start traversing the tree\n");
  int k = 0;
  double start_time = get_time();
/* Translated from #pragma omp target data ... */
{
    int32_t __arg_num = 7;
    int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 35};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(struct Node ) * no_of_nodes)), ((int64_t )(sizeof(_Bool ) * no_of_nodes)), ((int64_t )(sizeof(_Bool ) * no_of_nodes)), ((int64_t )(sizeof(_Bool ) * no_of_nodes)), ((int64_t )(sizeof(int ) * edge_list_size)), ((int64_t )(sizeof(int ) * no_of_nodes))};
    void *__args[] = {&no_of_nodes, h_graph_nodes + 0, h_graph_mask + 0, h_updating_graph_mask + 0, h_graph_visited + 0, h_graph_edges + 0, h_cost + 0};
    void *__args_base[] = {&no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    _Bool stop;
    do {
// if no thread changes this value then the loop stops
      stop = 0;
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = omp_num_threads;
        int _num_blocks_ = omp_num_teams;
        void *__host_ptr = (void *)(&OUT__1__4476__BFSGraph__135__id__);
        void *__args_base[] = {&no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost};
        void *__args[] = {&no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(struct Node *))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *)))};
        int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33};
        int32_t __arg_num = 7;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = omp_num_threads;
        int _num_blocks_ = omp_num_teams;
        void *__host_ptr = (void *)(&OUT__2__4476__BFSGraph__153__id__);
        void *__args_base[] = {&no_of_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, &stop};
        void *__args[] = {&no_of_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, &stop};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(_Bool *))), ((int64_t )(sizeof(_Bool )))};
        int64_t __arg_types[] = {33, 33, 33, 33, 35};
        int32_t __arg_num = 5;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
      k++;
    }while (stop);
    double end_time = get_time();
    printf("Compute time: %lfs\n",end_time - start_time);
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
// Store the result into a file
  FILE *fpo = fopen("result.log","w");
  for (int i = 0; i < no_of_nodes; i++) 
    fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
  fclose(fpo);
  printf("Result stored in result.log\n");
// cleanup memory
  free(h_graph_nodes);
  free(h_graph_edges);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
}
