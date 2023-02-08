#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"
#include "rex_nvidia.h" 
void run(int argc,char **argv);
/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)
#define BENCH_PRINT
#define NUM_TEAMS 256
#define NUM_THREADS 1024
extern int rows;
extern int cols;
extern int *data;
extern int *result;
#define M_SEED 9
void init(int ,char **);
void fatal(char *);
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
int main(int ,char **);
void run(int ,char **);
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__1__5826__run__89__kernel___exec_mode = 0;

__global__ void OUT__1__5826__run__89__kernel__(int *colsp__,int *tp__,int *_dev_data,int *_dev_src,int *_dev_dst)
{
  int n;
  typedef int int64_t;
  int _p_min;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *colsp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *colsp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (n = _dev_lower; n <= _dev_upper; n += 1) {
        _p_min = _dev_src[n];
        if (n > 0) 
          _p_min = (_p_min <= _dev_src[n - 1]?_p_min : _dev_src[n - 1]);
        if (n <  *colsp__ - 1) 
          _p_min = (_p_min <= _dev_src[n + 1]?_p_min : _dev_src[n + 1]);
        _dev_dst[n] = _dev_data[( *tp__ + 1) *  *colsp__ + n] + _p_min;
      }
  }
}
#ifdef __cplusplus
}
#endif
