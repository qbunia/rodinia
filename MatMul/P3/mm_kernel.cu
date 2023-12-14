#include "mm_cuda.h"
#include <stdio.h>
#define BLOCK_SIZE 16

#define BLOCK_SIZE 16 // Define an appropriate block size

__global__
void shared_block(REAL* A, REAL* B, REAL* C, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = n * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + n - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * n;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done
        // before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Store the result in the global memory matrix C
    int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + n * ty + tx] = Csub;
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
    shared_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);

    cudaMemcpy(C, C_device, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}
