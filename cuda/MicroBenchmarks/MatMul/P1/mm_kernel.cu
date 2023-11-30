#include "mm_cuda.h"
#include <stdio.h>
#define BLOCK_SIZE 16

__global__
void global_element(REAL* A, REAL* B, REAL* C, int n) {

    REAL C_value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int k = 0; k < n; k++) {
        C_value += A[row * n + k] * B[n * k + col];
    }

    // Each thread writes one element to C matrix
    C[row * n + col] = C_value;
}

void mm_kernel(REAL* A, REAL* B, REAL* C, int n, int kernel) {
    REAL *A_device, *B_device, *C_device;
    cudaMalloc(&A_device, n*n*sizeof(REAL));
    cudaMalloc(&B_device, n*n*sizeof(REAL));
    cudaMalloc(&C_device, n*n*sizeof(REAL));

    cudaMemcpy(A_device, A, n*n*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, n*n*sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);
    global_element<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);

    cudaMemcpy(C, C_device, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}
