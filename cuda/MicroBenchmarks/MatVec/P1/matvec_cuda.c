// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "matvec.h"

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL double
#define VEC_LEN 20480 //use a fixed number for now
//#define VEC_LEN_ 1024000 //use a fixed number for now
/* zero out the entire vector */
void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init_vector(REAL *vector, int m)
{
	for (int i = 0; i<m; i++) {
    vector[i] = (REAL)drand48();//(float)rand()/(float)(RAND_MAX/10.0);
  }
}

void init_matrix(REAL *matrix, int n, int m) {
	for (int i = 0; i<n; i++) {
		for (int j = 0; j<m; j++) {
			matrix[i*m+j] = (REAL)drand48();//(float)rand()/(float)(RAND_MAX/10.0);
		}
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

//serial version
void matvec_serial(int N, REAL *A, REAL *B, REAL *C) {
    int i,j;
    REAL temp;
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}

int main(int argc, char *argv[])
{
  int n,m;
  REAL *result, *vector, *matrix;
  REAL *res_serial;

  n = VEC_LEN;
  m = VEC_LEN;
  fprintf(stderr, "Usage: MatVec <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  result = (REAL *) malloc(n * sizeof(REAL));
  vector  = (REAL *) malloc(m * sizeof(REAL));
  matrix = (REAL *) malloc(n * m * sizeof(REAL));
  res_serial = (REAL *) malloc(n * sizeof(REAL));


  srand48(1<<12);
  init_vector(vector, m);
  init_matrix(matrix, n, m);

  int i;
  int num_runs = 5;
  
  for (i=0; i<num_runs; i++) matvec_serial(n, matrix, vector, res_serial);
  
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) matvec_cuda(result, vector, matrix, n, m);
  elapsed = (read_timer_ms() - elapsed)/num_runs;
  
  REAL checkresult = check(res_serial, result, n);
  printf("MatVec(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);

  //for(int i = 0; i<VEC_LEN; i++) printf("result[%d]:%f\n", i, result[i]);

  free(result);
  free(vector);
  free(matrix);
  return 0;
}
