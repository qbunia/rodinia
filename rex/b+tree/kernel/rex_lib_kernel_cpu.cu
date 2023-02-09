//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200
#define NUM_TEAMS 256
#define NUM_THREADS 1024
//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150
#include <omp.h>   // (in directory known to compiler)			needed by openmp
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
#include <stdlib.h> // (in directory known to compiler)			needed by malloc
//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150
#include "../common.h" // (in directory provided here)
//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150
#include "../util/timer/timer.h" // (in directory provided here)
//========================================================================================================================================================================================================200
//	KERNEL_CPU FUNCTION
//========================================================================================================================================================================================================200
#include "rex_nvidia.h" 
void kernel_cpu(int ,record *,knode *,long ,int ,long ,int ,long *,long *,int *,record *);
//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__1__6171__kernel_cpu__70__kernel___exec_mode = 0;

__global__ void OUT__1__6171__kernel_cpu__70__kernel__(record *records,long *knodes_elemp__,long *maxheightp__,int *countp__,int *threadsPerBlockp__,struct knode *_dev_knodes,long *_dev_currKnode,long *_dev_offset,int *_dev_keys,struct record *_dev_ans)
{
  typedef int int64_t;
  int _p_thid;
  int _p_bid;
  int _p_i;
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
// if value is between the two keys
            if (_dev_knodes[_dev_currKnode[_p_bid]] . keys[_p_thid] <= _dev_keys[_p_bid] && _dev_knodes[_dev_currKnode[_p_bid]] . keys[_p_thid + 1] > _dev_keys[_p_bid]) {
// this conditional statement is inserted to avoid crush due to but in
// original code "offset[bid]" calculated below that addresses
// knodes[] in the next iteration goes outside of its bounds cause
// segmentation fault more specifically, values saved into
// knodes->indices in the main function are out of bounds of knodes
// that they address
              if (((long )_dev_knodes[_dev_offset[_p_bid]] . indices[_p_thid]) <  *knodes_elemp__) {
                _dev_offset[_p_bid] = ((long )_dev_knodes[_dev_offset[_p_bid]] . indices[_p_thid]);
              }
            }
          }
// set for next tree level
          _dev_currKnode[_p_bid] = _dev_offset[_p_bid];
        }
// At this point, we have a candidate leaf node which may contain
// the target record.  Check each key to hopefully find the record
//  process all leaves at each level
        for (_p_thid = 0; _p_thid <  *threadsPerBlockp__; _p_thid++) {
          if (_dev_knodes[_dev_currKnode[_p_bid]] . keys[_p_thid] == _dev_keys[_p_bid]) {
            _dev_ans[_p_bid] . value = records[_dev_knodes[_dev_currKnode[_p_bid]] . indices[_p_thid]] . value;
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
