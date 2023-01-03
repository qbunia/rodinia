#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define STR_SIZE 256
/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
/* chip parameters	*/
#include "rex_nvidia.h" 
extern double t_chip;
extern double chip_height;
extern double chip_width;
/* ambient temperature, assuming no package at all	*/
__device__ double amb_temp;
/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(double *,double *,double *,int ,int ,double ,double ,double ,double ,double );
/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(double *,int ,double *,double *,int ,int );
void fatal(const char *);
void read_input(double *,int ,int ,char *);
void usage(int ,char **);
int main(int ,char **);
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__2__5416__single_iteration__112__kernel___exec_mode = 0;

__global__ void OUT__2__5416__single_iteration__112__kernel__(int *rowp__,int *colp__,double *_dev_result,double *_dev_temp)
{
  typedef int int64_t;
  int _p_r;
  int _p_c;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *rowp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *rowp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_r = _dev_lower; _p_r <= _dev_upper; _p_r += 1) {
        for (_p_c = 0; _p_c <  *colp__; _p_c++) {
          _dev_temp[_p_r *  *colp__ + _p_c] = _dev_result[_p_r *  *colp__ + _p_c];
        }
      }
  }
}
__device__ char OUT__1__5416__single_iteration__32__kernel___exec_mode = 0;

__global__ void OUT__1__5416__single_iteration__32__kernel__(int *rowp__,int *colp__,double *Capp__,double *Rxp__,double *Ryp__,double *Rzp__,double *stepp__,double *deltap__,double *_dev_result,double *_dev_temp,double *_dev_power)
{
  typedef int int64_t;
  int _p_r;
  int _p_c;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *rowp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *rowp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_r = _dev_lower; _p_r <= _dev_upper; _p_r += 1) {
        for (_p_c = 0; _p_c <  *colp__; _p_c++) {
/*	Corner 1	*/
          if (_p_r == 0 && _p_c == 0) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[0] + (_dev_temp[1] - _dev_temp[0]) /  *Rxp__ + (_dev_temp[ *colp__] - _dev_temp[0]) /  *Ryp__ + (amb_temp - _dev_temp[0]) /  *Rzp__);
/*	Corner 2	*/
          }
           else if (_p_r == 0 && _p_c ==  *colp__ - 1) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_c] + (_dev_temp[_p_c - 1] - _dev_temp[_p_c]) /  *Rxp__ + (_dev_temp[_p_c +  *colp__] - _dev_temp[_p_c]) /  *Ryp__ + (amb_temp - _dev_temp[_p_c]) /  *Rzp__);
/*	Corner 3	*/
          }
           else if (_p_r ==  *rowp__ - 1 && _p_c ==  *colp__ - 1) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__ + _p_c] + (_dev_temp[_p_r *  *colp__ + _p_c - 1] - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rxp__ + (_dev_temp[(_p_r - 1) *  *colp__ + _p_c] - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Ryp__ + (amb_temp - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rzp__);
/*	Corner 4	*/
          }
           else if (_p_r ==  *rowp__ - 1 && _p_c == 0) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__] + (_dev_temp[_p_r *  *colp__ + 1] - _dev_temp[_p_r *  *colp__]) /  *Rxp__ + (_dev_temp[(_p_r - 1) *  *colp__] - _dev_temp[_p_r *  *colp__]) /  *Ryp__ + (amb_temp - _dev_temp[_p_r *  *colp__]) /  *Rzp__);
/*	Edge 1	*/
          }
           else if (_p_r == 0) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_c] + (_dev_temp[_p_c + 1] + _dev_temp[_p_c - 1] - 2.0 * _dev_temp[_p_c]) /  *Rxp__ + (_dev_temp[ *colp__ + _p_c] - _dev_temp[_p_c]) /  *Ryp__ + (amb_temp - _dev_temp[_p_c]) /  *Rzp__);
/*	Edge 2	*/
          }
           else if (_p_c ==  *colp__ - 1) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__ + _p_c] + (_dev_temp[(_p_r + 1) *  *colp__ + _p_c] + _dev_temp[(_p_r - 1) *  *colp__ + _p_c] - 2.0 * _dev_temp[_p_r *  *colp__ + _p_c]) /  *Ryp__ + (_dev_temp[_p_r *  *colp__ + _p_c - 1] - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rxp__ + (amb_temp - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rzp__);
/*	Edge 3	*/
          }
           else if (_p_r ==  *rowp__ - 1) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__ + _p_c] + (_dev_temp[_p_r *  *colp__ + _p_c + 1] + _dev_temp[_p_r *  *colp__ + _p_c - 1] - 2.0 * _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rxp__ + (_dev_temp[(_p_r - 1) *  *colp__ + _p_c] - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Ryp__ + (amb_temp - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rzp__);
/*	Edge 4	*/
          }
           else if (_p_c == 0) {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__] + (_dev_temp[(_p_r + 1) *  *colp__] + _dev_temp[(_p_r - 1) *  *colp__] - 2.0 * _dev_temp[_p_r *  *colp__]) /  *Ryp__ + (_dev_temp[_p_r *  *colp__ + 1] - _dev_temp[_p_r *  *colp__]) /  *Rxp__ + (amb_temp - _dev_temp[_p_r *  *colp__]) /  *Rzp__);
/*	Inside the chip	*/
          }
           else {
             *deltap__ =  *stepp__ /  *Capp__ * (_dev_power[_p_r *  *colp__ + _p_c] + (_dev_temp[(_p_r + 1) *  *colp__ + _p_c] + _dev_temp[(_p_r - 1) *  *colp__ + _p_c] - 2.0 * _dev_temp[_p_r *  *colp__ + _p_c]) /  *Ryp__ + (_dev_temp[_p_r *  *colp__ + _p_c + 1] + _dev_temp[_p_r *  *colp__ + _p_c - 1] - 2.0 * _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rxp__ + (amb_temp - _dev_temp[_p_r *  *colp__ + _p_c]) /  *Rzp__);
          }
/*	Update Temperatures	*/
          _dev_result[_p_r *  *colp__ + _p_c] = _dev_temp[_p_r *  *colp__ + _p_c] +  *deltap__;
        }
      }
  }
}
#ifdef __cplusplus
}
#endif
