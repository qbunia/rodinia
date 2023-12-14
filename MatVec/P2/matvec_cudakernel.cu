#include "matvec.h"

__global__ 
void matvec_P2(REAL* matrix, REAL* vector, REAL* result, int n, int m) {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    
    int elementsPerBlock = (n + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < elementsPerBlock; i++) {
        int rowIndex = threadId + i * blockDim.x;
        
        if (rowIndex < n) {
            REAL temp = 0.0;
            for (int j = 0; j < m; j++) {
                temp += matrix[rowIndex * m + j] * vector[j];
            }
            result[rowIndex] = temp;
        }
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

  int blockSize = 1024;
  int gridSize = (m + blockSize - 1) / blockSize;  // Adjusted gridSize based on 'm'
  // Perform matvec elements
  matvec_P2<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, n, m);

  cudaMemcpy(result, d_result, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_result);
}
