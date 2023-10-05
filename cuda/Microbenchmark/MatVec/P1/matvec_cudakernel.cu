#include "matvec.h"

__global__ void matvec_P1(REAL* matrix, REAL* vector, REAL* result, int n, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        REAL temp = 0.0;
        for (int j = 0; j < m; j++)
            temp += matrix[i * m + j] * vector[j];
        result[i] = temp;
    }
}

void matvec_cuda(REAL* result, REAL* vector, REAL* matrix, int n, int m) {
  REAL *d_matrix, *d_vector, *d_result;
  cudaMalloc(&d_matrix, n*m*sizeof(REAL));
  cudaMalloc(&d_vector, m*sizeof(REAL));
  cudaMalloc(&d_result, n*sizeof(REAL));

  cudaMemcpy(d_matrix, matrix, n*m*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, vector, m*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, result, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform matvec elements
  matvec_P1<<<(n+255)/256, 256>>>(d_matrix, d_vector, d_result, n, m);

  cudaMemcpy(result, d_result, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_result);
}
