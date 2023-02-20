#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024
#define MAX_ARGS 10
#define REC_LENGTH 49   // size of a record in db
#define REC_WINDOW 10   // number of records to read at a time
#define LATITUDE_POS 28 // location of latitude coordinates in input record
#define OPEN 10000      // initial value of nearest neighbors
#include "rex_nvidia.h" 

struct neighbor 
{
  char entry[49];
  double dist;
}
;
typedef struct latLong {
float lat;
float lng;}LatLong;
/**
 * This program finds the k-nearest neighbors
 * Usage:	./nn <filelist> <num> <target latitude> <target longitude>
 *			filelist: File with the filenames to the records
 *			num: Number of nearest neighbors to find
 *			target lat: Latitude coordinate for distance
 *calculations target long: Longitude coordinate for distance calculations The
 *filelist and data are generated by hurricane_gen.c REC_WINDOW has been
 *arbitrarily assigned; A larger value would allow more work for the threads
 */
int main(int ,char *[]);
#ifdef __cplusplus
extern "C" {
#endif
__device__ char OUT__1__4144__main__121__kernel___exec_mode = 0;

__global__ void OUT__1__4144__main__121__kernel__(int rec_countp__2,float target_latp__2,float target_longp__2,struct latLong *_dev_locations,float *_dev_z)
{
  int *rec_countp__ = &rec_countp__2;
  float *target_latp__ = &target_latp__2;
  float *target_longp__ = &target_longp__2;

  int _p_i;
{
    int _dev_lower;
    int _dev_upper;
    int _dev_loop_chunk_size;
    int _dev_loop_sched_index;
    int _dev_loop_stride;
    int _dev_thread_num = getCUDABlockThreadCount(1);
    int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
    XOMP_static_sched_init(0, *rec_countp__ - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
    while(XOMP_static_sched_next(&_dev_loop_sched_index, *rec_countp__ - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
      for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
        _dev_z[_p_i] = ((float )(sqrt((double )((_dev_locations[_p_i] . lat -  *target_latp__) * (_dev_locations[_p_i] . lat -  *target_latp__) + (_dev_locations[_p_i] . lng -  *target_longp__) * (_dev_locations[_p_i] . lng -  *target_longp__)))));
      }
  }
}
#ifdef __cplusplus
}
#endif
