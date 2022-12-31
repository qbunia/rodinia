#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
#include "rex_nvidia.h" 
extern ::FILE *fp;
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
__device__ char OUT__2__5397__BFSGraph__151__kernel___exec_mode = 0;

#ifdef __cplusplus
extern "C" {
#endif
__global__ void OUT__2__5397__BFSGraph__151__kernel__(int *no_of_nodesp__,bool *stopp__,bool *_dev_h_graph_mask,bool *_dev_h_updating_graph_mask,bool *_dev_h_graph_visited)
{
  int tid;
  typedef int int64_t;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0,*no_of_nodesp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index,*no_of_nodesp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (tid = _dev_lower; tid <= _dev_upper; tid += 1) {
        if (((int )_dev_h_updating_graph_mask[tid]) == ((int )true)) {
          _dev_h_graph_mask[tid] = true;
          _dev_h_graph_visited[tid] = true;
          *stopp__ = true;
          _dev_h_updating_graph_mask[tid] = false;
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
__device__ char OUT__1__5397__BFSGraph__133__kernel___exec_mode = 0;

#ifdef __cplusplus
extern "C" {
#endif
__global__ void OUT__1__5397__BFSGraph__133__kernel__(int *no_of_nodesp__,struct Node *_dev_h_graph_nodes,bool *_dev_h_graph_mask,bool *_dev_h_updating_graph_mask,bool *_dev_h_graph_visited,int *_dev_h_graph_edges,int *_dev_h_cost)
{
  int tid;
  typedef int int64_t;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0,*no_of_nodesp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index,*no_of_nodesp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (tid = _dev_lower; tid <= _dev_upper; tid += 1) {
        if (((int )_dev_h_graph_mask[tid]) == ((int )true)) {
          _dev_h_graph_mask[tid] = false;
          for (int i = _dev_h_graph_nodes[tid] . starting; i < _dev_h_graph_nodes[tid] . no_of_edges + _dev_h_graph_nodes[tid] . starting; i++) {
            int id = _dev_h_graph_edges[i];
            if (!_dev_h_graph_visited[id]) {
              _dev_h_cost[id] = _dev_h_cost[tid] + 1;
              _dev_h_updating_graph_mask[id] = true;
            }
          }
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
