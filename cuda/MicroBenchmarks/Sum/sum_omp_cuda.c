// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "sum_omp_cuda.h"
#define TEST 20

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL double
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
        A[i] = (REAL)drand48();
    }
}

/*serial version */
REAL sum_serial(REAL* input, int n) {
    int i;
    REAL sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int n = 512;
    int kernel = 0;
    REAL *output_device, *output, *input, sum;

    if (argc > 2) {
        n = atoi(argv[1]);
        kernel = atoi(argv[2]);
    }
    else {
        printf("=================================\n");
        printf("Usage: ./sum <n> <kernel>\n");
        printf("Default size: n = 512, kernel = 0\n");
        printf("Kernel -1: test all kernels\n");
        printf("Kernel 0: P1 level optimization\n");
        printf("Kernel 1: P1 level optimization - shared memory\n");
        printf("=================================\n");
    };
    output_device = (REAL *) malloc(n * sizeof(REAL));
    output = (REAL *) malloc(n * sizeof(REAL));
    input = (REAL *) malloc(n * sizeof(REAL));

    srand48(1<<12);
    init(input, n);

    REAL res_serial = sum_serial(input, n);


    int i;

    REAL res_cuda;

    double elapsed = read_timer_ms();
    for (i = 0; i < TEST; i++) {
        res_cuda = sum_kernel(input, n, kernel);
    };
    elapsed = (read_timer_ms() - elapsed)/TEST;
    printf("GPU kernel %d: %g\n", kernel, elapsed);
    printf("CUDA: %g\n", res_cuda);

    REAL checkresult = res_serial - res_cuda;
    printf("sum(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);

    free(output_device);
    free(output);
    free(input);
    return 0;
}
