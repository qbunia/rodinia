#include "stencil_omp_cuda.h"
#include <stdio.h>
#define BLOCK_SIZE 16

//Each threac computes one pixel and pixels are first read from global memory to the shared memory by each thread, then reading the pixel to do the stencil 
//are from shared memory. By using shared memory, we take advantage that there is re-use locality of pixel by multiple threads 
__global__
void shared_element(REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int p = threadIdx.y;
    int q = threadIdx.x;

    __shared__ REAL sdata[BLOCK_SIZE*BLOCK_SIZE];
    sdata[p*BLOCK_SIZE + q] = src[i*width + j];
    __syncthreads();

    REAL sum = 0;
    #pragma unroll
    for (int n = 0; n < flt_width; n++) {
        for (int m = 0; m < flt_height; m++) {
            int x = j + n - flt_width / 2;
            int y = i + m - flt_height / 2;
            if (x >= 0 && x < width && y >= 0 && y < height) {
                int idx = m*flt_width + n;
                int x2 = q + n - flt_width / 2;
                int y2 = p + m - flt_height / 2;
                // if the pixel has been loaded into shared memory, read from there
                // otherwise read from global memory
                if (x2 >= 0 && x2 < BLOCK_SIZE && y2 >= 0 && y2 < BLOCK_SIZE) {
                    sum += sdata[y2*BLOCK_SIZE + x2] * filter[idx];
                }
                else {
                    sum += src[y*width + x] * filter[idx]; //block boundary, load neighbor pixel from global memory
                };
            }
        }
    }

    // Each thread writes one element to C matrix
    dst[i*width + j] = sum;
}

void stencil_kernel(REAL* input, REAL* output, int width, int height, const float* filter, int filter_width, int filter_height, int kernel) {
    REAL *input_device, *output_device;
    float *filter_device;
    cudaMalloc(&input_device, width*height*sizeof(REAL));
    cudaMalloc(&output_device, width*height*sizeof(REAL));
    cudaMalloc(&filter_device, filter_width*filter_height*sizeof(float));

    cudaMemcpy(input_device, input, width*height*sizeof(REAL), cudaMemcpyHostToDevice);
    //cudaMemcpy(output_device, output, width*height*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_device, filter, filter_width*filter_height*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    shared_element<<<dimGrid, dimBlock>>>(input_device, output_device, width, height, filter_device, filter_width, filter_height);

    cudaMemcpy(output, output_device, width*height*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(input_device);
    cudaFree(output_device);
    cudaFree(filter_device);
}
