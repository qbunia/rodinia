#include "axpy.h"

__global__ 
void
axpy_cudakernel_warmingup(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}

__global__ 
void
axpy_cudakernel_P1(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}

void axpy_kernel(int N, REAL* Y, REAL* X, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, N*sizeof(REAL));
  cudaMalloc(&d_y, N*sizeof(REAL));

  cudaMemcpy(d_x, X, N*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, Y, N*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_warmingup<<<(N+255)/256, 256>>>(d_x, d_y, N, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_P1<<<(N+255)/256, 256>>>(d_x, d_y, N, a);
  cudaDeviceSynchronize();

  cudaMemcpy(Y, d_y, N*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

