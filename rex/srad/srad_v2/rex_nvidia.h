#include <stdint.h>

extern __device__ void XOMP_static_sched_init(int lb, int up, int step, int orig_chunk_size, int _p_num_threads, int _p_thread_id, \
              int * loop_chunk_size, int * loop_sched_index, int * loop_stride);
extern __device__ int XOMP_static_sched_next(
    int* loop_sched_index , int loop_end, int orig_step, int loop_stride, int loop_chunk_size,
    int _p_num_threads, int _p_thread_id,
    int *lb,int *ub);
extern __device__ int getCUDABlockThreadCount(int dimension_no);
extern __device__ int getLoopIndexFromCUDAVariables(int dimension_no);

struct DeviceEnvironmentTy {
  uint32_t DebugKind;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
};
