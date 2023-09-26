/*
 * AXPY  Y[N] = Y[N] + a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
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

void axpy_omp_Offloading_P0(int N, REAL *Y, REAL *X, REAL a);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> ");
        exit(1);
    }
    N = atoi(argv[1]);

    REAL a = 123.456;
    REAL *Y_base = malloc(sizeof(REAL)*N);
    REAL *X = malloc(sizeof(REAL)* N);

    srand48((1 << 12));
    init(X, N);
    init(Y_base, N);

    int i;
    int num_runs = 10;
    
    //Warm-up
    axpy_omp_Offloading_P0(N, Y_base, X, a);
    
    double elapsed_omp_P0 = read_timer();
    for (i=0; i<num_runs; i++) axpy_omp_Offloading_P0(N, Y_base, X, a);
    elapsed_omp_P0 = (read_timer() - elapsed_omp_P0)/num_runs;
    
    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \t\t\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("axpy_omp_Offloading_P0:\t\t%4f\t%4f \t\n", elapsed_omp_P0 * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_omp_P0));
    free(Y_base);
    free(X);

    return 0;
}

void axpy_omp_Offloading_P0(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}