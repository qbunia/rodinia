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

int main(int argc, char *argv[]) {
   if (argc < 3) {
       fprintf(stderr, "Usage: %s <serial/omp_cpu/omp_gpu/CUDA> <kernel_number> <n> <num_threads>\n", argv[0]);
       exit(1);
   }
   int default_paralle_model = 0;
   int default_kernel_number = 0;
   int default_N = 102400;
   int default_num_threads = 4;
    
   int paralle_model = (argc > 1) ? atoi(argv[1]) : default_paralle_model;
   int kernel_number = (argc > 2) ? atoi(argv[2]) : default_kernel_number;
   int N = (argc > 3) ? atoi(argv[3]) : default_N;
   int num_threads = (argc > 4) ? atoi(argv[4]) : default_num_threads;
   
   omp_set_num_threads(num_threads);
   REAL a = 123.456;
   REAL *X = malloc(sizeof(REAL) * N);
   REAL *Y_base = malloc(sizeof(REAL) * N);
   REAL *Y_parallel = malloc(sizeof(REAL) * N);

   srand48((1 << 12));
   init(X, N);
   init(Y_base, N);
   memcpy(Y_parallel, Y_base, N * sizeof(REAL));

   int i;
   int num_runs = 10;
   
   for (i=0; i<num_runs+1; i++) axpy_P0(N, Y_base, X, a);
       
   printf("======================================================================================================\n");
   printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d threads for dist\n", N, num_threads);
   printf("------------------------------------------------------------------------------------------------------\n");
   char device_type[10];
   
   double elapsed_omp_parallel_for = 0.0;

   if (paralle_model == 0) {
       strcpy(device_type, "serial");
       elapsed_omp_parallel_for = read_timer();
       for (i=0; i<num_runs+1; i++) axpy_P0(N, Y_parallel, X, a);
       elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
   } else if (paralle_model == 1) {
       strcpy(device_type, "omp_cpu");

       switch (kernel_number) {
           case 1:
               elapsed_omp_parallel_for = read_timer();
               for (i=0; i<num_runs+1; i++) axpy_omp_P1(N, Y_parallel, X, a);
               elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
               break;
           case 2:
               elapsed_omp_parallel_for = read_timer();
               for (i=0; i<num_runs+1; i++) axpy_omp_P2(N, Y_parallel, X, a);
               elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
               break;
           case 3:
               elapsed_omp_parallel_for = read_timer();
               for (i=0; i<num_runs+1; i++) axpy_omp_P3(N, Y_parallel, X, a);
               elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
               break;
           default:
               fprintf(stderr, "Invalid.\n");
               exit(1);
       }
   } else if(paralle_model == 2){
       strcpy(device_type, "omp_gpu");
       elapsed_omp_parallel_for = read_timer();
       for (i=0; i<num_runs+1; i++) axpy_omp_offloading(N, Y_parallel, X, a);
       elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
   } else if(paralle_model == 3){
       *strcpy(device_type, "cuda");
       switch (kernel_number) {
           case 1:
               printf("P1: \n");
               elapsed_omp_parallel_for = read_timer();
               for (i=0; i<num_runs+1; i++) axpy_P0(N, Y_parallel, X, a);
               elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
               break;
           case 2:
               elapsed_omp_parallel_for = read_timer();
               for (i=0; i<num_runs+1; i++) axpy_P0(N, Y_parallel, X, a);
               elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for)/num_runs;
               break;
           default:
               fprintf(stderr, "Invalid.\n");
               exit(1);
       }
   }

   printf("------------------------------------------------------------------------------------------------------\n");
   printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
   printf("------------------------------------------------------------------------------------------------------\n");
   printf("axpy_%s_P%d:\t\t%4f\t%4f \t\t%g\n", device_type, kernel_number, elapsed_omp_parallel_for * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_omp_parallel_for), check(Y_base,Y_parallel, N));

   free(Y_base);
   free(Y_parallel);
   free(X);

   return 0;
}
