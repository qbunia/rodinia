#include "rex_kmp.h" 
char OUT__1__6316__kernel_cpu_2__80__id__ = 0;
struct __tgt_offload_entry OUT__1__6316__kernel_cpu_2__80__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__6316__kernel_cpu_2__80__id__)), "OUT__1__6316__kernel_cpu_2__80__kernel__", 0, 0, 0};
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

void kernel_cpu_2(int cores_arg,knode *knodes,long knodes_elem,int order,long maxheight,int count,long *currKnode,long *offset,long *lastKnode,long *offset_2,int *start,int *end,int *recstart,int *reclength)
{
//======================================================================================================================================================150
//	Variables
//======================================================================================================================================================150
// timer
  long long time0;
  long long time1;
  long long time2;
// common variables
  int i;
  time0 = get_time();
//======================================================================================================================================================150
//	MCPU SETUP
//======================================================================================================================================================150
  int threadsPerBlock;
  threadsPerBlock = (order < 1024?order : 1024);
  time1 = get_time();
//======================================================================================================================================================150
//	PROCESS INTERACTIONS
//======================================================================================================================================================150
// private thread IDs
  int thid;
  int bid;
{
/* Launch CUDA kernel ... */
    int64_t __device_id = 0;
    int _threads_per_block_ = 1024;
    int _num_blocks_ = 256;
    void *__host_ptr = (void *)(&OUT__1__6316__kernel_cpu_2__80__id__);
    void *__args_base[] = {knodes, &knodes_elem, &maxheight, &count, &threadsPerBlock, currKnode, offset, lastKnode, offset_2, start, end, recstart, reclength};
    void *__args[] = {knodes, &knodes_elem, &maxheight, &count, &threadsPerBlock, currKnode + 0, offset + 0, lastKnode + 0, offset_2 + 0, start + 0, end + 0, recstart + 0, reclength + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(knode *))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(long ) * count)), ((int64_t )(sizeof(long ) * count)), ((int64_t )(sizeof(long ) * count)), ((int64_t )(sizeof(long ) * count)), ((int64_t )(sizeof(int ) * count)), ((int64_t )(sizeof(int ) * count)), ((int64_t )(sizeof(int ) * count)), ((int64_t )(sizeof(int ) * count))};
    int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 35, 35};
    int32_t __arg_num = 13;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
  time2 = get_time();
//======================================================================================================================================================150
//	DISPLAY TIMING
//======================================================================================================================================================150
  printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
  printf("%15.12f s, %15.12f %% : MCPU: SET DEVICE\n",(((float )(time1 - time0)) / 1000000),(((float )(time1 - time0)) / ((float )(time2 - time0)) * 100));
  printf("%15.12f s, %15.12f %% : CPU/MCPU: KERNEL\n",(((float )(time2 - time1)) / 1000000),(((float )(time2 - time1)) / ((float )(time2 - time0)) * 100));
  printf("Total time:\n");
  printf("%.12f s\n",(((float )(time2 - time0)) / 1000000));
// main
}