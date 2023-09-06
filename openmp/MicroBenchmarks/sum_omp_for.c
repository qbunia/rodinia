/*
 * Sum of a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <omp.h>

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

#ifdef LOG_RESULTS
void log_result(char *algo, int numbers, int threads, double elapsed)
{
    FILE* f = fopen("results.txt", "a");
    fprintf(f, "%s,%d,%d,%f\n", algo, numbers, threads, elapsed * 1.0e3);
    fclose(f);
}
#else
#define log_result(algo, numbers, threads, elapsed)
#endif

#define REAL float
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

int num_threads = 2;
REAL sum(int N, REAL X[], REAL a);
REAL sum_omp_parallel_for(int N, REAL X[], REAL a);
REAL sum_omp_parallel_optimization(int N, REAL X[], REAL a);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    if (argc < 2) {
        fprintf(stderr, "Usage: sum <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc >=3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    
    REAL *X = malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(X, N);
    REAL a = 0.1234;
    /* example run */
    double serial_elapsed = read_timer();
    REAL serial_result = sum(N, X, a);
    serial_elapsed = (read_timer() - serial_elapsed);

    double parallel_for_elapsed = read_timer();
    REAL parallel_for_result = sum_omp_parallel_for(N, X, a);
    parallel_for_elapsed = (read_timer() - parallel_for_elapsed);

    double parallel_elapsed = read_timer();
    REAL parallel_elapsed_result = sum_omp_parallel_optimization(N, X, a);
    parallel_elapsed = (read_timer() - parallel_elapsed);

    log_result("serial", N, num_threads, serial_elapsed);
    log_result("parallel for", N, num_threads, parallel_for_elapsed);
    log_result("parallel", N, num_threads, parallel_elapsed);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tSum %d numbers using OpenMP with %d threads\n", N, num_threads);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum serial:      \t%4f\t%4f\n", serial_elapsed * 1.0e3, 2*N / (1.0e6 * serial_elapsed));
    printf("Sum parallel for:\t%4f\t%4f\n", parallel_for_elapsed * 1.0e3, 2*N / (1.0e6 * parallel_for_elapsed));
   // printf("Sum parallel:    \t%4f\t%4f\n", parallel_elapsed * 1.0e3, 2*N / (1.0e6 * parallel_elapsed));

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

/*
 * Parallel reduction using OpenMP parallel for and reduction clause
 */
REAL sum_omp_parallel_for(int N, REAL X[], REAL a) {
    int i;
    REAL result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < N; ++i)
    {
        result += a * X[i];
    }
    return result;
}

/*
 * Optimized reduction using only OpenMP parallel, and you should implement
 * your own way of reduction
 */
REAL sum_omp_parallel_optimization(int N, REAL X[], REAL a) {

    REAL total = 0;

    #pragma omp parallel
    {
        int chunk_size = N / num_threads;
        int thread_num = omp_get_thread_num();

        int start = thread_num*chunk_size;
        int stop = (thread_num+1)*chunk_size;

        if (thread_num == num_threads-1)
        {
            int remaining = N % num_threads;
            stop += remaining;
        }

        REAL local_result = 0.0;
        int i;
        for (i = start; i < stop; ++i)
        {
            local_result += a * X[i];
        }

        #pragma omp critical
        {
            total += local_result;
        }
    }

    return total;
}
