#include "rex_kmp.h" 
char OUT__1__5826__run__89__id__ = 0;
struct __tgt_offload_entry OUT__1__5826__run__89__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__5826__run__89__id__)), "OUT__1__5826__run__89__kernel__", 0, 0, 0};
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"
void run(int argc,char **argv);
/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)
#define BENCH_PRINT
#define NUM_TEAMS 256
#define NUM_THREADS 1024
int rows;
int cols;
int *data;
int *result;
#define M_SEED 9

void init(int argc,char **argv)
{
  if (argc == 3) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
  }
   else {
    printf("Usage: pathfiner width num_of_steps\n");
    exit(0);
  }
  data = ((int *)(malloc((rows * cols) * sizeof(int ))));
  result = ((int *)(malloc(cols * sizeof(int ))));
  int seed = 9;
  srand(seed);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data[i * cols + j] = rand() % 10;
    }
  }
  for (int j = 0; j < cols; j++) 
    result[j] = data[j];
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ",data[i * cols + j]);
    }
    printf("\n");
  }
#endif
}

void fatal(char *s)
{
  fprintf(stderr,"error: %s\n",s);
}
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

int main(int argc,char **argv)
{
  int status = 0;
  run(argc,argv);
  return 0;
}

void run(int argc,char **argv)
{
  init(argc,argv);
  unsigned long long cycles;
  int *src;
  int *dst;
  int *temp;
  int min;
  dst = result;
  src = ((int *)(malloc(cols * sizeof(int ))));
  start_cycles = (rdtsc());
/* Translated from #pragma omp target data ... */
{
    int32_t __arg_num = 3;
    int64_t __arg_types[] = {33, 35, 35};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ) * (rows * cols))), ((int64_t )(sizeof(int ) * cols)), ((int64_t )(sizeof(int ) * cols))};
    void *__args[] = {data + 0, src + 0, dst + 0};
    void *__args_base[] = {data, src, dst};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    for (int t = 0; t < rows - 1; t++) {
      temp = src;
      src = dst;
      dst = temp;
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = 1024;
        int _num_blocks_ = 256;
        void *__host_ptr = (void *)(&OUT__1__5826__run__89__id__);
        void *__args_base[] = {&cols, &t, data, src, dst};
        void *__args[] = {&cols, &t, data + 0, src + 0, dst + 0};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ) * (rows * cols))), ((int64_t )(sizeof(int ) * cols)), ((int64_t )(sizeof(int ) * cols))};
        int64_t __arg_types[] = {33, 33, 33, 35, 35};
        int32_t __arg_num = 5;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
    }
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
  cycles = (rdtsc()) - start_cycles;
  printf("timer: %Lu\n",cycles);
#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++) 
    printf("%d ",data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++) 
    printf("%d ",dst[i]);
  printf("\n");
#endif
  free(data);
  free(dst);
  free(src);
}
