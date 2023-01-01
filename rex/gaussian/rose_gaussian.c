#include "rex_kmp.h" 
char OUT__3__5564__Fan2__196__id__ = 0;
struct __tgt_offload_entry OUT__3__5564__Fan2__196__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__3__5564__Fan2__196__id__)), "OUT__3__5564__Fan2__196__kernel__", 0, 0, 0};
char OUT__2__5564__Fan2__188__id__ = 0;
struct __tgt_offload_entry OUT__2__5564__Fan2__188__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__2__5564__Fan2__188__id__)), "OUT__2__5564__Fan2__188__kernel__", 0, 0, 0};
char OUT__1__5564__Fan1__174__id__ = 0;
struct __tgt_offload_entry OUT__1__5564__Fan1__174__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__5564__Fan1__174__id__)), "OUT__1__5564__Fan1__174__kernel__", 0, 0, 0};
/*-----------------------------------------------------------
 ** gaussian.c -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.
 **   The sequential version is gaussian.c.  This parallel
 **   implementation converts three independent for() loops
 **   into three Fans.  Use the data file ge_3.dat to verify
 **   the correction of the output.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 ** Modified by Pisit Makpaisit for OpenACC, 08/05/2013
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
int Size;
float *a;
float *b;
float *finalVec;
float *m;
FILE *fp;
void InitProblemOnce(char *filename);
void InitPerRun(float *m);
void ForwardSub();
void BackSub();
void Fan1(float *m,float *a,int Size,int t);
void Fan2(float *m,float *a,float *b,int Size,int j1,int t);
void InitMat(float *ary,int nrow,int ncol);
void InitAry(float *ary,int ary_size);
void PrintMat(float *ary,int nrow,int ncolumn);
void PrintAry(float *ary,int ary_size);
unsigned int totalKernelTime = 0;

int main(int argc,char *argv[])
{
  int status = 0;
  struct timeval time_start;
  struct timeval time_end;
  unsigned int time_total;
  int verbose = 1;
  if (argc < 2) {
    printf("Usage: gaussian matrix.txt [-q]\n\n");
    printf("-q (quiet) suppresses printing the matrix and result values.\n");
    printf("The first line of the file contains the dimension of the matrix, n.");
    printf("The second line of the file is a newline.\n");
    printf("The next n lines contain n tab separated values for the matrix.");
    printf("The next line of the file is a newline.\n");
    printf("The next line of the file is a 1xn vector with tab separated values.\n");
    printf("The next line of the file is a newline. (optional)\n");
    printf("The final line of the file is the pre-computed solution. (optional)\n");
    printf("Example: matrix4.txt:\n");
    printf("4\n");
    printf("\n");
    printf("-0.6\t-0.5\t0.7\t0.3\n");
    printf("-0.3\t-0.9\t0.3\t0.7\n");
    printf("-0.4\t-0.5\t-0.3\t-0.8\n");
    printf("0.0\t-0.1\t0.2\t0.9\n");
    printf("\n");
    printf("-0.85\t-0.68\t0.24\t-0.53\n");
    printf("\n");
    printf("0.7\t0.0\t-0.4\t-0.5\n");
    exit(0);
  }
// char filename[100];
// sprintf(filename,"matrices/matrix%d.txt",size);
  InitProblemOnce(argv[1]);
  if (argc > 2) {
    if (!strcmp(argv[2],"-q")) 
      verbose = 0;
  }
// InitProblemOnce(filename);
  InitPerRun(m);
// begin timing
  gettimeofday(&time_start,(void *)0);
// run kernels
  ForwardSub();
// end timing
  gettimeofday(&time_end,(void *)0);
  time_total = (time_end . tv_sec * 1000000 + time_end . tv_usec - (time_start . tv_sec * 1000000 + time_start . tv_usec));
  if (verbose) {
    printf("Matrix m is: \n");
    PrintMat(m,Size,Size);
    printf("Matrix a is: \n");
    PrintMat(a,Size,Size);
    printf("Array b is: \n");
    PrintAry(b,Size);
  }
  BackSub();
  if (verbose) {
    printf("The final solution is: \n");
    PrintAry(finalVec,Size);
  }
  printf("\nTime total (including memory transfers)\t%f sec\n",time_total * 1e-6);
  printf("Time for kernels:\t%f sec\n",totalKernelTime * 1e-6);
/*printf("%d,%d\n",size,time_total);
  fprintf(stderr,"%d,%d\n",size,time_total);*/
  free(m);
  free(a);
  free(b);
}
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */

void InitProblemOnce(char *filename)
{
// char *filename = argv[1];
// printf("Enter the data file name: ");
// scanf("%s", filename);
// printf("The file name is: %s\n", filename);
  fp = fopen(filename,"r");
  fscanf(fp,"%d",&Size);
  a = ((float *)(malloc((Size * Size) * sizeof(float ))));
  InitMat(a,Size,Size);
// printf("The input matrix a is:\n");
// PrintMat(a, Size, Size);
  b = ((float *)(malloc(Size * sizeof(float ))));
  InitAry(b,Size);
// printf("The input array b is:\n");
// PrintAry(b, Size);
  m = ((float *)(malloc((Size * Size) * sizeof(float ))));
}
/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */

void InitPerRun(float *m)
{
  int i;
  for (i = 0; i < Size * Size; i++) 
     *(m + i) = 0.0;
}
/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */

void Fan1(float *m,float *a,int Size,int t)
{
  int i;
{
/* Launch CUDA kernel ... */
    int _threads_per_block_ = 128;
    int _num_blocks_ = 256;
    int64_t __device_id = 0;
    void *__host_ptr = (void *)(&OUT__1__5564__Fan1__174__id__);
    void *__args_base[] = {&Size, &t, m, a};
    void *__args[] = {&Size, &t, m + 0, a + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(float ) * (Size * Size))), ((int64_t )(sizeof(float ) * (Size * Size)))};
    int64_t __arg_types[] = {33, 33, 35, 35};
    int32_t __arg_num = 4;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
}
/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

void Fan2(float *m,float *a,float *b,int Size,int j1,int t)
{
  int i;
  int j;
{
/* Launch CUDA kernel ... */
    int _threads_per_block_ = 128;
    int _num_blocks_ = 256;
    int64_t __device_id = 0;
    void *__host_ptr = (void *)(&OUT__2__5564__Fan2__188__id__);
    void *__args_base[] = {&Size, &t, m, a};
    void *__args[] = {&Size, &t, m + 0, a + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(float ) * (Size * Size))), ((int64_t )(sizeof(float ) * (Size * Size)))};
    int64_t __arg_types[] = {33, 33, 35, 35};
    int32_t __arg_num = 4;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
{
/* Launch CUDA kernel ... */
    int _threads_per_block_ = 128;
    int _num_blocks_ = 256;
    int64_t __device_id = 0;
    void *__host_ptr = (void *)(&OUT__3__5564__Fan2__196__id__);
    void *__args_base[] = {&Size, &t, m, b};
    void *__args[] = {&Size, &t, m + 0, b + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(float ) * (Size * Size))), ((int64_t )(sizeof(float ) * Size))};
    int64_t __arg_types[] = {33, 33, 35, 35};
    int32_t __arg_num = 4;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
}
/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */

void ForwardSub()
{
  int t;
/* Translated from #pragma omp target data ... */
{
    int32_t __arg_num = 3;
    int64_t __arg_types[] = {35, 35, 35};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(float ) * (Size * Size))), ((int64_t )(sizeof(float ) * Size)), ((int64_t )(sizeof(float ) * (Size * Size)))};
    void *__args[] = {a + 0, b + 0, m + 0};
    void *__args_base[] = {a, b, m};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    struct timeval time_start;
    gettimeofday(&time_start,(void *)0);
    for (t = 0; t < Size - 1; t++) {
      Fan1(m,a,Size,t);
      Fan2(m,a,b,Size,Size - t,t);
    }
// end timing kernels
    struct timeval time_end;
    gettimeofday(&time_end,(void *)0);
    totalKernelTime = (time_end . tv_sec * 1000000 + time_end . tv_usec - (time_start . tv_sec * 1000000 + time_start . tv_usec));
/* end omp target data */
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
}
/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
// create a new vector to hold the final answer
  finalVec = ((float *)(malloc(Size * sizeof(float ))));
// solve "bottom up"
  int i;
  int j;
  for (i = 0; i < Size; i++) {
    finalVec[Size - i - 1] = b[Size - i - 1];
    for (j = 0; j < i; j++) {
      finalVec[Size - i - 1] -=  *(a + Size * (Size - i - 1) + (Size - j - 1)) * finalVec[Size - j - 1];
    }
    finalVec[Size - i - 1] = finalVec[Size - i - 1] /  *(a + Size * (Size - i - 1) + (Size - i - 1));
  }
}

void InitMat(float *ary,int nrow,int ncol)
{
  int i;
  int j;
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      fscanf(fp,"%f",ary + Size * i + j);
    }
  }
}
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */

void PrintMat(float *ary,int nrow,int ncol)
{
  int i;
  int j;
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      printf("%8.2f ",( *(ary + Size * i + j)));
    }
    printf("\n");
  }
  printf("\n");
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */

void InitAry(float *ary,int ary_size)
{
  int i;
  for (i = 0; i < ary_size; i++) {
    fscanf(fp,"%f",&ary[i]);
  }
}
/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */

void PrintAry(float *ary,int ary_size)
{
  int i;
  for (i = 0; i < ary_size; i++) {
    printf("%.2f ",ary[i]);
  }
  printf("\n\n");
}
