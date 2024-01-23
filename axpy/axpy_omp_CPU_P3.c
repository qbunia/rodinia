#include "axpy.h"
#include <omp.h>

void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    int i;
#pragma omp parallel shared(N, X, Y, a) private(i)
    {
#pragma omp for simd
        for (i = 0; i < N; ++i)
            Y[i] += a * X[i];
    }
}
