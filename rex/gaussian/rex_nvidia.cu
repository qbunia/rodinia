#include "rex_nvidia.h"

#include "xomp_cuda_lib.cu"
#include "xomp_cuda_lib_inlined.cu"

__device__ DeviceEnvironmentTy omptarget_device_environment = {0, 0, 0, 0};
