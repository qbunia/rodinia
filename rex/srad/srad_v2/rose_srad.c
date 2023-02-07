#include "rex_kmp.h" 
char OUT__2__5292__main__157__id__ = 0;
struct __tgt_offload_entry OUT__2__5292__main__157__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__2__5292__main__157__id__)), "OUT__2__5292__main__157__kernel__", 0, 0, 0};
char OUT__1__5292__main__120__id__ = 0;
struct __tgt_offload_entry OUT__1__5292__main__120__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__5292__main__120__id__)), "OUT__1__5292__main__120__kernel__", 0, 0, 0};
// srad.cpp : Defines the entry point for the console application.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
void random_matrix(float *I,int rows,int cols);

void usage(int argc,char **argv)
{
  fprintf(stderr,"Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads><lamda> <no. of iter>\n",argv[0]);
  fprintf(stderr,"\t<rows>   - number of rows\n");
  fprintf(stderr,"\t<cols>    - number of cols\n");
  fprintf(stderr,"\t<y1> \t - y1 value of the speckle\n");
  fprintf(stderr,"\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr,"\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr,"\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr,"\t<no. of threads>  - no. of threads\n");
  fprintf(stderr,"\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr,"\t<no. of iter>   - number of iterations\n");
  exit(1);
}

int main(int argc,char *argv[])
{
  int status = 0;
  int rows = 0;
  int cols = 0;
  int size_I;
  int size_R;
  int niter = 10;
  int iter;
  int k;
  float *I;
  float *J;
  float q0sqr;
  float sum;
  float sum2;
  float tmp;
  float meanROI;
  float varROI;
  float Jc;
  float G2;
  float L;
  float num;
  float den;
  float qsqr;
  int *iN;
  int *iS;
  int *jE;
  int *jW;
  float *dN;
  float *dS;
  float *dW;
  float *dE;
  int r1 = 0;
  int r2 = 0;
  int c1 = 0;
  int c2 = 0;
  float cN;
  float cS;
  float cW;
  float cE;
  float *c;
  float D;
  float lambda;
  int i;
  int j;
  if (argc == 9) {
    rows = atoi(argv[1]);
// number of rows in the domain
    cols = atoi(argv[2]);
// number of cols in the domain
    if (rows % 16 != 0 || cols % 16 != 0) {
      fprintf(stderr,"rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1 = atoi(argv[3]);
// y1 position of the speckle
    r2 = atoi(argv[4]);
// y2 position of the speckle
    c1 = atoi(argv[5]);
// x1 position of the speckle
    c2 = atoi(argv[6]);
// x2 position of the speckle
    lambda = (atof(argv[7]));
// Lambda value
    niter = atoi(argv[8]);
// number of iterations
  }
   else {
    usage(argc,argv);
  }
  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);
  I = ((float *)(malloc(size_I * sizeof(float ))));
  J = ((float *)(malloc(size_I * sizeof(float ))));
  c = ((float *)(malloc(sizeof(float ) * size_I)));
  iN = ((int *)(malloc(sizeof(unsigned int *) * rows)));
  iS = ((int *)(malloc(sizeof(unsigned int *) * rows)));
  jW = ((int *)(malloc(sizeof(unsigned int *) * cols)));
  jE = ((int *)(malloc(sizeof(unsigned int *) * cols)));
  dN = ((float *)(malloc(sizeof(float ) * size_I)));
  dS = ((float *)(malloc(sizeof(float ) * size_I)));
  dW = ((float *)(malloc(sizeof(float ) * size_I)));
  dE = ((float *)(malloc(sizeof(float ) * size_I)));
  for (int i = 0; i < rows; i++) {
    iN[i] = i - 1;
    iS[i] = i + 1;
  }
  for (int j = 0; j < cols; j++) {
    jW[j] = j - 1;
    jE[j] = j + 1;
  }
  iN[0] = 0;
  iS[rows - 1] = rows - 1;
  jW[0] = 0;
  jE[cols - 1] = cols - 1;
  printf("Randomizing the input matrix\n");
  random_matrix(I,rows,cols);
  for (k = 0; k < size_I; k++) {
    J[k] = ((float )(exp(I[k])));
  }
  printf("Start the SRAD main loop\n");
/* Translated from #pragma omp target data ... */
{
    int32_t __arg_num = 10;
    int64_t __arg_types[] = {35, 33, 33, 33, 33, 33, 33, 33, 33, 33};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(float ) * size_I)), ((int64_t )(sizeof(int ) * rows)), ((int64_t )(sizeof(int ) * rows)), ((int64_t )(sizeof(int ) * cols)), ((int64_t )(sizeof(int ) * cols)), ((int64_t )(sizeof(float ) * size_I)), ((int64_t )(sizeof(float ) * size_I)), ((int64_t )(sizeof(float ) * size_I)), ((int64_t )(sizeof(float ) * size_I)), ((int64_t )(sizeof(float ) * size_I))};
    void *__args[] = {J + 0, iN + 0, iS + 0, jE + 0, jW + 0, dN + 0, dS + 0, dW + 0, dE + 0, c + 0};
    void *__args_base[] = {J, iN, iS, jE, jW, dN, dS, dW, dE, c};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    for (iter = 0; iter < niter; iter++) {
      sum = 0;
      sum2 = 0;
      for (i = r1; i <= r2; i++) {
        for (j = c1; j <= c2; j++) {
          tmp = J[i * cols + j];
          sum += tmp;
          sum2 += tmp * tmp;
        }
      }
      meanROI = sum / size_R;
      varROI = sum2 / size_R - meanROI * meanROI;
      q0sqr = varROI / (meanROI * meanROI);
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = 1024;
        int _num_blocks_ = 256;
        void *__host_ptr = (void *)(&OUT__1__5292__main__120__id__);
        void *__args_base[] = {&rows, &cols, J, &q0sqr, iN, iS, jE, jW, dN, dS, dW, dE, c};
        void *__args[] = {&rows, &cols, J, &q0sqr, iN, iS, jE, jW, dN, dS, dW, dE, c};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float ))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *)))};
        int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33};
        int32_t __arg_num = 13;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = 1024;
        int _num_blocks_ = 256;
        void *__host_ptr = (void *)(&OUT__2__5292__main__157__id__);
        void *__args_base[] = {&rows, &cols, J, iS, jE, dN, dS, dW, dE, c, &lambda};
        void *__args[] = {&rows, &cols, J, iS, jE, dN, dS, dW, dE, c, &lambda};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float )))};
        int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33};
        int32_t __arg_num = 11;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
    }
// iterations end
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
// target data region ends
#ifdef OUTPUT
#endif
  printf("Computation Done\n");
  free(I);
  free(J);
  free(iN);
  free(iS);
  free(jW);
  free(jE);
  free(dN);
  free(dS);
  free(dW);
  free(dE);
  free(c);
  return 0;
}

void random_matrix(float *I,int rows,int cols)
{
  srand(7);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = (rand()) / ((float )2147483647);
    }
  }
}
