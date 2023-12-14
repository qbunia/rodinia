#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include <omp.h>
#define TEAM_NUM 1024
#define TEAM_SIZE 256

#define REAL double

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL double
#define VEC_LEN 1024000 //use a fixed number for now
/* zero out the entire vector */
void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = (double)drand48();
    }
}

/*serial version */
void axpy_base(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

/*openmp offloading */
void axpy_omp_offloading_P0(REAL* x, REAL* y, long n, REAL a) {
  int i;
  #pragma omp target teams distribute parallel for map(to: n, x[0:n]) map(tofrom: y[0:n]) num_teams(TEAM_NUM) num_threads(TEAM_SIZE)
  for (i = 0; i < n; ++i)
  {
    y[i] += a * x[i];
  }
}

/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, int n)
{
    int i;
    REAL diffsum =0.0, sum = 0.0;
    for (i = 0; i < n; i++) {
        diffsum += fabs(A[i] - B[i]);
        sum += fabs(B[i]);
    }
    return diffsum/sum;
}

int main(int argc, char *argv[])
{
  int n;
  REAL *y, *x, *y_base;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  y = (REAL *) malloc(n * sizeof(REAL));
  y_base = (REAL *) malloc(n * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));
  


  srand48(1<<12);
  init(x, n);
  init(y_base, n);
  memcpy(y, y_base, n * sizeof(REAL));

  int i;
  int num_runs = 10;
  
  for (i=0; i<num_runs+1; i++) axpy_base(n, y_base, x, a);

  //warm up
  axpy_omp_offloading_P0(x, y, n, a);

  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy_omp_offloading_P0(x, y, n, a);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  printf("axpy(%d): time: %0.2fms\n", n, elapsed);
  
  double error = check(y_base,y, n);
  printf("error: %0.2fms\n", error);

  free(y);
  free(x);
  return 0;
}
