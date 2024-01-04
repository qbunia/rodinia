#include "axpy.h"
#include <omp.h>

void axpy_omp_P2(int N, REAL *Y, REAL *X, REAL a) {
    int i;
#pragma omp parallel shared(N, X, Y, a) private(i)
#pragma omp for schedule(guided, 64)
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
