#include "matvec.h"

__global__ void matvec_P4(REAL* matrix, REAL* vector, REAL* result, int n, int m)
{
    __shared__ float shared_x[64];
    int row = threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (n - 1) / 64 + 1;

    for (int i = 0; i < num_tiles; i++) {
        int tile_start = i * 64;
        int tile_end = min(tile_start + 64, n);

        if (tile_start + row < m) {
            shared_x[row] = vector[tile_start + row];
        }

        __syncthreads();

        for (int j = 0; j < 64; j++) {
            if (tile_start + j < tile_end && tile_start + row < m) {
                sum += matrix[(tile_start + row) * n + tile_start + j] * shared_x[j];
            }
        }

        __syncthreads();
    }

    if (64 * blockIdx.x + row < m) {
        result[64 * blockIdx.x + row] = sum;
    }
}

void matvec_cuda(REAL* result, REAL* vector, REAL* matrix, int n, int m) {
    REAL *d_matrix, *d_vector, *d_result;
    cudaMalloc(&d_matrix, n * m * sizeof(REAL));
    cudaMalloc(&d_vector, m * sizeof(REAL));
    cudaMalloc(&d_result, n * sizeof(REAL));

    cudaMemcpy(d_matrix, matrix, n * m * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, m * sizeof(REAL), cudaMemcpyHostToDevice);

    int blockSize = 64;
    int gridSize = (m + blockSize - 1) / blockSize;  // Adjusted gridSize based on 'm'

    matvec_P4<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, n, m);

    cudaMemcpy(result, d_result, n * sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}
