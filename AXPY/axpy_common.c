#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "axpy.h"

#define REAL float
#define VECTOR_LENGTH 102400

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double)tm.time + (double)tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double)tm.time * 1000.0 + (double)tm.millitm;
}

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double)drand48();
    }
}

double check(REAL *A, REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}
