#include "mm_cuda.h"
#include <stdio.h>
#define BLOCK_SIZE 16

__global__
void global_block(REAL* A, REAL* B, REAL* C, int n) {
    int wA = n;
    int wB = n;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += A[a + wA * ty + k] * B[b + wB * k + tx];
        }
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
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
    global_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);

    cudaMemcpy(C, C_device, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}
