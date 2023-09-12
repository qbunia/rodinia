// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

//#include <omp.h>
#define TEAM_NUM 1024
#define TEAM_SIZE 1024

static double read_timer_ms() {
    struct timeval timer;
    gettimeofday(&timer, NULL);
    return (double)timer.tv_sec * 1000.0 + (double)timer.tv_usec / 1000.0;
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
void axpy(REAL* x, REAL* y, long n, REAL a) {
  //int i;
  #pragma acc parallel loop copyin(a, n, x[0:n]) copyout(y[0:n]) num_gangs(1) num_workers(1) vector_length(TEAM_SIZE)
  for (int i = 0; i < n; ++i)
  {
    y[i] += a * x[i];
  }
}

/* compare two arrays and return percentage of difference */
/*
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
*/

int main(int argc, char *argv[])
{
  int n;
  REAL *y, *x;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  y  = (REAL *) malloc(n * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  init(y, n);

  int i;
  int num_runs = 10;
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy(x, y, n, a);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  printf("axpy(%d): time: %0.2fms\n", n, elapsed);
  //assert (checkresult < 1.0e-10);

  free(y);
  free(x);
  return 0;
}
