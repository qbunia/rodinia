#include "axpy.h"

int main(int argc, char *argv[]) {
   if (argc < 3) {
       fprintf(stderr, "Usage: %s <n> <num_threads> [full_report]\n", argv[0]);
       exit(1);
   }

   int default_N = 1024000;
   int default_num_threads = 4;
   int default_full_report = 1;

   int N = atoi(argv[1]);
   int num_threads = atoi(argv[2]);
   int full_report = (argc > 3) ? atoi(argv[3]) : default_full_report;

   
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
   
   //for (i=0; i<num_runs+1; i++) axpy_P0(N, Y_base, X, a);
       

   char device_type[10];
   
   double elapsed_omp_parallel_for = read_timer();

   axpy_kernel(N, Y_parallel, X, a);
    
   elapsed_omp_parallel_for = (read_timer() - elapsed_omp_parallel_for);
   if(full_report == 1){
       printf("======================================================================================================\n");
       printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d threads for dist\n", N, num_threads);
       printf("------------------------------------------------------------------------------------------------------\n");
       printf("------------------------------------------------------------------------------------------------------\n");
       printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
       printf("------------------------------------------------------------------------------------------------------\n");
       printf("axpy_kernel:\t\t%4f\t%4f \t\t%g\n", elapsed_omp_parallel_for * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_omp_parallel_for), check(Y_base,Y_parallel, N));
   } else {
        printf("%4f",elapsed_omp_parallel_for);
   }

   free(Y_base);
   free(Y_parallel);
   free(X);

   return 0;
}
