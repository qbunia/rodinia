#include "axpy.h"
#include <omp.h>

/*openmp offloading */
void axpy_omp_offloading(int N, REAL* Y, REAL* X, REAL a) {
  int i;
  #pragma omp target teams distribute parallel for map(to: N, X[0:N]) map(tofrom: Y[0:N]) num_teams(TEAM_NUM) num_threads(TEAM_SIZE)
  for (i = 0; i < N; ++i)
  {
    Y[i] += a * X[i];
  }
}
