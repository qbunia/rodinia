#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <openacc.h>

/* read timer in second */
double read_timer() {
    struct timeval timer;
    gettimeofday(&timer, NULL);
    return ((double)timer.tv_sec + (double)timer.tv_usec / 1000000.0);
}

#define REAL float
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (REAL)drand48();
    }
}

REAL sum(int N, REAL X[], REAL a);
REAL sum_acc_parallel(int N, REAL X[], REAL a);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_threads = 2;
    if (argc < 2) {
        fprintf(stderr, "Usage: sum <n> (default %d)\n", N);
        exit(1);
    }
    N = atoi(argv[1]);

    REAL *X = (REAL *)malloc(sizeof(REAL) * N);

    srand48((1 << 12));
    init(X, N);
    REAL a = 0.1234;

    double serial_elapsed = read_timer();
    REAL serial_result = sum(N, X, a);
    serial_elapsed = (read_timer() - serial_elapsed);

    double acc_elapsed = read_timer();
    REAL acc_result = sum_acc_parallel(N, X, a);
    acc_elapsed = (read_timer() - acc_elapsed);

    printf("======================================================================================================\n");
    printf("\tSum %d numbers using OpenACC\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum serial:      \t%4f\t%4f\n", serial_elapsed * 1.0e3, 2 * N / (1.0e6 * serial_elapsed));
    printf("Sum OpenACC:     \t%4f\t%4f\n", acc_elapsed * 1.0e3, 2 * N / (1.0e6 * acc_elapsed));

    free(X);
    return 0;
}

REAL sum(int N, REAL X[], REAL a) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i)
        result += a * X[i];
    return result;
}

REAL sum_acc_parallel(int N, REAL X[], REAL a) {
    int i;
    REAL result = 0.0;

    #pragma acc parallel loop reduction(+:result)
    for (i = 0; i < N; ++i) {
        result += a * X[i];
    }

    return result;
}
