//====================================================================================================100
//		UPDATE
//====================================================================================================100
//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments
//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"
#define NUM_TEAMS 256
#define NUM_THREADS 1024
//====================================================================================================100
//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100
//====================================================================================================100
#include "rex_nvidia.h" 
int main(int ,char *[]);
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__2__5286__main__295__kernel___exec_mode = 0;

__global__ void OUT__2__5286__main__295__kernel__(float *image,long *Nrp__,long *Ncp__,float *lambdap__,int *iS,int *jE,float *dN,float *dS,float *dW,float *dE,float *c)
{
  float _p_D;
  float _p_cN;
  float _p_cS;
  float _p_cW;
  float _p_cE;
  long _p_i;
  long _p_j;
  long _p_k;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init((long )0, *Ncp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *Ncp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_j = _dev_lower; _p_j <= _dev_upper; _p_j += 1) {
// do for the range of columns in IMAGE
        for (_p_i = ((long )0); _p_i <  *Nrp__; _p_i++) {
// do for the range of rows in IMAGE
// current index
          _p_k = _p_i +  *Nrp__ * _p_j;
// get position of current element
// diffusion coefficent
          _p_cN = c[_p_k];
// north diffusion coefficient
          _p_cS = c[((long )iS[_p_i]) +  *Nrp__ * _p_j];
// south diffusion coefficient
          _p_cW = c[_p_k];
// west diffusion coefficient
          _p_cE = c[_p_i +  *Nrp__ * ((long )jE[_p_j])];
// east diffusion coefficient
// divergence (equ 58)
          _p_D = _p_cN * dN[_p_k] + _p_cS * dS[_p_k] + _p_cW * dW[_p_k] + _p_cE * dE[_p_k];
// divergence
// image update (equ 61) (every element of IMAGE)
          image[_p_k] = ((float )(((double )image[_p_k]) + 0.25 * ((double )( *lambdap__)) * ((double )_p_D)));
// updates image (based on input time step and divergence)
        }
      }
  }
}
__device__ char OUT__1__5286__main__244__kernel___exec_mode = 0;

__global__ void OUT__1__5286__main__244__kernel__(float *image,long *Nrp__,long *Ncp__,float *q0sqrp__,int *iN,int *iS,int *jE,int *jW,float *dN,float *dS,float *dW,float *dE,float *c)
{
  float _p_Jc;
  float _p_G2;
  float _p_L;
  float _p_num;
  float _p_den;
  float _p_qsqr;
  long _p_i;
  long _p_j;
  long _p_k;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init((long )0, *Ncp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *Ncp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_j = _dev_lower; _p_j <= _dev_upper; _p_j += 1) {
// do for the range of columns in IMAGE
        for (_p_i = ((long )0); _p_i <  *Nrp__; _p_i++) {
// do for the range of rows in IMAGE
// current index/pixel
          _p_k = _p_i +  *Nrp__ * _p_j;
// get position of current element
          _p_Jc = image[_p_k];
// get value of the current element
// directional derivates (every element of IMAGE)
          dN[_p_k] = image[((long )iN[_p_i]) +  *Nrp__ * _p_j] - _p_Jc;
// north direction derivative
          dS[_p_k] = image[((long )iS[_p_i]) +  *Nrp__ * _p_j] - _p_Jc;
// south direction derivative
          dW[_p_k] = image[_p_i +  *Nrp__ * ((long )jW[_p_j])] - _p_Jc;
// west direction derivative
          dE[_p_k] = image[_p_i +  *Nrp__ * ((long )jE[_p_j])] - _p_Jc;
// east direction derivative
// normalized discrete gradient mag squared (equ 52,53)
          _p_G2 = (dN[_p_k] * dN[_p_k] + dS[_p_k] * dS[_p_k] + dW[_p_k] * dW[_p_k] + dE[_p_k] * dE[_p_k]) / (_p_Jc * _p_Jc);
// gradient (based on derivatives)
// normalized discrete laplacian (equ 54)
          _p_L = (dN[_p_k] + dS[_p_k] + dW[_p_k] + dE[_p_k]) / _p_Jc;
// laplacian (based on derivatives)
// ICOV (equ 31/35)
          _p_num = ((float )(0.5 * ((double )_p_G2) - 1.0 / 16.0 * ((double )(_p_L * _p_L))));
// num (based on gradient and laplacian)
          _p_den = ((float )(((double )1) + .25 * ((double )_p_L)));
// den (based on laplacian)
          _p_qsqr = _p_num / (_p_den * _p_den);
// qsqr (based on num and den)
// diffusion coefficent (equ 33) (every element of IMAGE)
          _p_den = (_p_qsqr -  *q0sqrp__) / ( *q0sqrp__ * (((float )1) +  *q0sqrp__));
// den (based on qsqr and q0sqr)
          c[_p_k] = ((float )(1.0 / (1.0 + ((double )_p_den))));
// diffusion coefficient (based on den)
// saturate diffusion coefficent to 0-1 range
          if (c[_p_k] < ((float )0)) 
// if diffusion coefficient < 0
{
            c[_p_k] = ((float )0);
          }
           else 
// ... set to 0
if (c[_p_k] > ((float )1)) 
// if diffusion coefficient > 1
{
            c[_p_k] = ((float )1);
// ... set to 1
          }
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
