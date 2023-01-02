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
#include "rex_nvidia.h" 
extern int Size;
extern float *a;
extern float *b;
extern float *finalVec;
extern float *m;
extern FILE *fp;
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
extern unsigned int totalKernelTime;
int main(int ,char *[]);
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *);
/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(float *);
/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
void Fan1(float *,float *,int ,int );
/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
void Fan2(float *,float *,float *,int ,int ,int );
/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub();
/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */
void BackSub();
void InitMat(float *,int ,int );
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *,int ,int );
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *,int );
/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *,int );
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__3__5564__Fan2__196__kernel___exec_mode = 0;

__global__ void OUT__3__5564__Fan2__196__kernel__(int *Sizep__,int *tp__,float *_dev_m,float *_dev_b)
{
  typedef int int64_t;
  int _p_i;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *Sizep__ - 1 -  *tp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *Sizep__ - 1 -  *tp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
        _dev_b[_p_i + 1 +  *tp__] -= _dev_m[ *Sizep__ * (_p_i + 1 +  *tp__) +  *tp__] * _dev_b[ *tp__];
      }
  }
}
__device__ char OUT__2__5564__Fan2__188__kernel___exec_mode = 0;

__global__ void OUT__2__5564__Fan2__188__kernel__(int *Sizep__,int *tp__,float *_dev_m,float *_dev_a)
{
  typedef int int64_t;
  int _p_i;
  int _p_j;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *Sizep__ - 1 -  *tp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *Sizep__ - 1 -  *tp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
        for (_p_j = 0; _p_j <  *Sizep__ -  *tp__; _p_j++) 
          _dev_a[ *Sizep__ * (_p_i + 1 +  *tp__) + (_p_j +  *tp__)] -= _dev_m[ *Sizep__ * (_p_i + 1 +  *tp__) +  *tp__] * _dev_a[ *Sizep__ *  *tp__ + (_p_j +  *tp__)];
      }
  }
}
__device__ char OUT__1__5564__Fan1__174__kernel___exec_mode = 0;

__global__ void OUT__1__5564__Fan1__174__kernel__(int *Sizep__,int *tp__,float *_dev_m,float *_dev_a)
{
  typedef int int64_t;
  int _p_i;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *Sizep__ - 1 -  *tp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *Sizep__ - 1 -  *tp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
        _dev_m[ *Sizep__ * (_p_i +  *tp__ + 1) +  *tp__] = _dev_a[ *Sizep__ * (_p_i +  *tp__ + 1) +  *tp__] / _dev_a[ *Sizep__ *  *tp__ +  *tp__];
      }
  }
}
#ifdef __cplusplus
}
#endif