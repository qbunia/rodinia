//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200
#define NUM_TEAMS 256
#define NUM_THREADS 1024
//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150
#include <stdio.h>  // (in directory known to compiler)
#include <stdlib.h> // (in directory known to compiler)
//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150
#include "../common.h" // (in directory provided here)
//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150
#include "../util/timer/timer.h" // (in directory provided here)	needed by timer
//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150
#include "./kernel_cpu_2.h" // (in directory provided here)
//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200
#include "rex_nvidia.h" 
void kernel_cpu_2(int ,knode *,long ,int ,long ,int ,long *,long *,long *,long *,int *,int *,int *,int *);
//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__1__6316__kernel_cpu_2__80__kernel___exec_mode = 0;

__global__ void OUT__1__6316__kernel_cpu_2__80__kernel__(knode *knodes,long *knodes_elemp__,long *maxheightp__,int *countp__,int *threadsPerBlockp__,long *_dev_currKnode,long *_dev_offset,long *_dev_lastKnode,long *_dev_offset_2,int *_dev_start,int *_dev_end,int *_dev_recstart,int *_dev_reclength)
{
  typedef int int64_t;
  int _p_i;
  int _p_thid;
  int _p_bid;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *countp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *countp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_bid = _dev_lower; _p_bid <= _dev_upper; _p_bid += 1) {
// process levels of the tree
        for (_p_i = 0; ((long )_p_i) <  *maxheightp__; _p_i++) {
// process all leaves at each level
          for (_p_thid = 0; _p_thid <  *threadsPerBlockp__; _p_thid++) {
            if (knodes[_dev_currKnode[_p_bid]] . keys[_p_thid] <= _dev_start[_p_bid] && knodes[_dev_currKnode[_p_bid]] . keys[_p_thid + 1] > _dev_start[_p_bid]) {
// this conditional statement is inserted to avoid crush due to but in
// original code "offset[bid]" calculated below that later addresses
// part of knodes goes outside of its bounds cause segmentation fault
// more specifically, values saved into knodes->indices in the main
// function are out of bounds of knodes that they address
              if (((long )knodes[_dev_currKnode[_p_bid]] . indices[_p_thid]) <  *knodes_elemp__) {
                _dev_offset[_p_bid] = ((long )knodes[_dev_currKnode[_p_bid]] . indices[_p_thid]);
              }
            }
            if (knodes[_dev_lastKnode[_p_bid]] . keys[_p_thid] <= _dev_end[_p_bid] && knodes[_dev_lastKnode[_p_bid]] . keys[_p_thid + 1] > _dev_end[_p_bid]) {
// this conditional statement is inserted to avoid crush due to but in
// original code "offset_2[bid]" calculated below that later addresses
// part of knodes goes outside of its bounds cause segmentation fault
// more specifically, values saved into knodes->indices in the main
// function are out of bounds of knodes that they address
              if (((long )knodes[_dev_lastKnode[_p_bid]] . indices[_p_thid]) <  *knodes_elemp__) {
                _dev_offset_2[_p_bid] = ((long )knodes[_dev_lastKnode[_p_bid]] . indices[_p_thid]);
              }
            }
          }
// set for next tree level
          _dev_currKnode[_p_bid] = _dev_offset[_p_bid];
          _dev_lastKnode[_p_bid] = _dev_offset_2[_p_bid];
        }
// process leaves
        for (_p_thid = 0; _p_thid <  *threadsPerBlockp__; _p_thid++) {
// Find the index of the starting record
          if (knodes[_dev_currKnode[_p_bid]] . keys[_p_thid] == _dev_start[_p_bid]) {
            _dev_recstart[_p_bid] = knodes[_dev_currKnode[_p_bid]] . indices[_p_thid];
          }
        }
// process leaves
        for (_p_thid = 0; _p_thid <  *threadsPerBlockp__; _p_thid++) {
// Find the index of the ending record
          if (knodes[_dev_lastKnode[_p_bid]] . keys[_p_thid] == _dev_end[_p_bid]) {
            _dev_reclength[_p_bid] = knodes[_dev_lastKnode[_p_bid]] . indices[_p_thid] - _dev_recstart[_p_bid] + 1;
          }
        }
      }
  }
}
//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
#ifdef __cplusplus
}
#endif
