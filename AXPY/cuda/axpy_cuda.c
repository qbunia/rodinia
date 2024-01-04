#include "../axpy.h"

int main(int argc, char *argv[])
{
  int n;
  REAL *y_cuda, *y, *x;
  REAL a = 123.456;

  n = VECTOR_LENGTH;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  y_cuda = (REAL *) malloc(n * sizeof(REAL));
  y  = (REAL *) malloc(n * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  init(y_cuda, n);
  memcpy(y, y_cuda, n*sizeof(REAL));

  int i;
  int num_runs = 10;
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy_cuda(x, y_cuda, n, a);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  REAL checkresult = check(y_cuda, y, n);
  printf("axpy(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);
  //assert (checkresult < 1.0e-10);

  free(y_cuda);
  free(y);
  free(x);
  return 0;
}
