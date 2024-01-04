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

void axpy_P0(int N, REAL *Y, REAL *X, REAL a);
void axpy_omp_P1(int N, REAL *Y, REAL *X, REAL a);
void axpy_omp_P2(int N, REAL *Y, REAL *X, REAL a);
void axpy_omp_P3(int N, REAL *Y, REAL *X, REAL a);
void axpy_omp_offloading(int N, REAL *Y, REAL *X, REAL a);

#endif

#ifdef __cplusplus
extern "C" {
#endif
extern void axpy_cuda(REAL *x, REAL * y, int n, REAL a);
#ifdef __cplusplus
}
#endif

