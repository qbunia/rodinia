#include "axpy.h"

void axpy_P0(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}
