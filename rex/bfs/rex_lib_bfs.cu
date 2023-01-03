#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
#include "rex_nvidia.h" 
extern FILE *fp;
// Structure to hold a node information

struct Node 
{
  int starting;
  int no_of_edges;
}
;
void BFSGraph(int argc,char **argv);
void Usage(int ,char **);
double get_time();
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int ,char **);
////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int ,char **);
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__2__4476__BFSGraph__153__kernel___exec_mode = 0;

__global__ void OUT__2__4476__BFSGraph__153__kernel__(int *no_of_nodesp__,_Bool *h_graph_mask,_Bool *h_updating_graph_mask,_Bool *h_graph_visited,_Bool *stopp__)
{
  int tid;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *no_of_nodesp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *no_of_nodesp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (tid = _dev_lower; tid <= _dev_upper; tid += 1) {
        if (((int )h_updating_graph_mask[tid]) == 1) {
          h_graph_mask[tid] = ((_Bool )1);
          h_graph_visited[tid] = ((_Bool )1);
           *stopp__ = ((_Bool )1);
          h_updating_graph_mask[tid] = ((_Bool )0);
        }
      }
  }
}
__device__ char OUT__1__4476__BFSGraph__135__kernel___exec_mode = 0;

__global__ void OUT__1__4476__BFSGraph__135__kernel__(int *no_of_nodesp__,struct Node *h_graph_nodes,_Bool *h_graph_mask,_Bool *h_updating_graph_mask,_Bool *h_graph_visited,int *h_graph_edges,int *h_cost)
{
  int tid;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *no_of_nodesp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *no_of_nodesp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (tid = _dev_lower; tid <= _dev_upper; tid += 1) {
        if (((int )h_graph_mask[tid]) == 1) {
          h_graph_mask[tid] = ((_Bool )0);
          for (int i = h_graph_nodes[tid] . starting; i < h_graph_nodes[tid] . no_of_edges + h_graph_nodes[tid] . starting; i++) {
            int id = h_graph_edges[i];
            if (!h_graph_visited[id]) {
              h_cost[id] = h_cost[tid] + 1;
              h_updating_graph_mask[id] = ((_Bool )1);
            }
          }
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
