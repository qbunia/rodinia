// axpy_common.h

#ifndef AXPY_H
#define AXPY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <omp.h>

/* read timer in second */
double read_timer();

/* read timer in ms */
double read_timer_ms();

#define REAL float
#define VECTOR_LENGTH 102400
#define TEAM_NUM 1024
#define TEAM_SIZE 256

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N);

double check(REAL *A, REAL B[], int N);

#ifdef __cplusplus
extern "C" {
#endif
extern void axpy_kernel(int N, REAL *Y, REAL *X, REAL a);
#ifdef __cplusplus
}
#endif

#endif
