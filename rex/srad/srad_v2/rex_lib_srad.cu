// srad.cpp : Defines the entry point for the console application.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
#include "rex_nvidia.h" 
void random_matrix(float *I,int rows,int cols);
void usage(int ,char **);
int main(int ,char *[]);
void random_matrix(float *,int ,int );
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__2__5292__main__157__kernel___exec_mode = 0;

__global__ void OUT__2__5292__main__157__kernel__(int *rowsp__,int *colsp__,float *J,int *iS,int *jE,float *dN,float *dS,float *dW,float *dE,float *c,float *lambdap__)
{
  int i;
  int _p_k;
  float _p_cN;
  float _p_cS;
  float _p_cW;
  float _p_cE;
  float _p_D;
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
    XOMP_static_sched_init(0, *rowsp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *rowsp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (i = _dev_lower; i <= _dev_upper; i += 1) {
        for (int j = 0; j <  *colsp__; j++) {
// current index
          _p_k = i *  *colsp__ + j;
// diffusion coefficent
          _p_cN = c[_p_k];
          _p_cS = c[iS[i] *  *colsp__ + j];
          _p_cW = c[_p_k];
          _p_cE = c[i *  *colsp__ + jE[j]];
// divergence (equ 58)
          _p_D = _p_cN * dN[_p_k] + _p_cS * dS[_p_k] + _p_cW * dW[_p_k] + _p_cE * dE[_p_k];
// image update (equ 61)
          J[_p_k] = ((float )(((double )J[_p_k]) + 0.25 * ((double )( *lambdap__)) * ((double )_p_D)));
        }
      }
  }
}
__device__ char OUT__1__5292__main__120__kernel___exec_mode = 0;

__global__ void OUT__1__5292__main__120__kernel__(int *rowsp__,int *colsp__,float *J,float *q0sqrp__,int *iN,int *iS,int *jE,int *jW,float *dN,float *dS,float *dW,float *dE,float *c)
{
  int i;
  int _p_k;
  float _p_Jc;
  float _p_G2;
  float _p_L;
  float _p_num;
  float _p_den;
  float _p_qsqr;
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
    XOMP_static_sched_init(0, *rowsp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *rowsp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (i = _dev_lower; i <= _dev_upper; i += 1) {
        for (int j = 0; j <  *colsp__; j++) {
          _p_k = i *  *colsp__ + j;
          _p_Jc = J[_p_k];
// directional derivates
          dN[_p_k] = J[iN[i] *  *colsp__ + j] - _p_Jc;
          dS[_p_k] = J[iS[i] *  *colsp__ + j] - _p_Jc;
          dW[_p_k] = J[i *  *colsp__ + jW[j]] - _p_Jc;
          dE[_p_k] = J[i *  *colsp__ + jE[j]] - _p_Jc;
          _p_G2 = (dN[_p_k] * dN[_p_k] + dS[_p_k] * dS[_p_k] + dW[_p_k] * dW[_p_k] + dE[_p_k] * dE[_p_k]) / (_p_Jc * _p_Jc);
          _p_L = (dN[_p_k] + dS[_p_k] + dW[_p_k] + dE[_p_k]) / _p_Jc;
          _p_num = ((float )(0.5 * ((double )_p_G2) - 1.0 / 16.0 * ((double )(_p_L * _p_L))));
          _p_den = ((float )(((double )1) + .25 * ((double )_p_L)));
          _p_qsqr = _p_num / (_p_den * _p_den);
// diffusion coefficent (equ 33)
          _p_den = (_p_qsqr -  *q0sqrp__) / ( *q0sqrp__ * (((float )1) +  *q0sqrp__));
          c[_p_k] = ((float )(1.0 / (1.0 + ((double )_p_den))));
// saturate diffusion coefficent
          if (c[_p_k] < ((float )0)) {
            c[_p_k] = ((float )0);
          }
           else if (c[_p_k] > ((float )1)) {
            c[_p_k] = ((float )1);
          }
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
