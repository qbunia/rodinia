/*
 * matrix vector
 * A[N][N] * B[N] = C[N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>


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

void init(int N, REAL *A) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
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

void matvec_base(int N, REAL *A, REAL *B, REAL *C) {
    int i,j;
    REAL temp;
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}

void matvec_P3(int N, REAL *A, REAL *B, REAL *C) {
    int i,j;
    REAL temp;
    #pragma omp parallel shared(N, A, B, C) private(i, j, temp)
    {
        #pragma omp for schedule(guided, 64)
        for (i = 0; i < N; i++) {
            temp = 0.0;

            #pragma omp simd reduction(+:temp)
            for (j = 0; j < N; j++) {
                temp += A[i * N + j] * B[j];
            }

            C[i] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    int N;

    int num_threads = 4; /* 4 is default number of threads */
    if (argc < 2) {
        fprintf(stderr, "Usage: mv <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc >=3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double elapsed_omp;

    REAL *A = malloc(sizeof(REAL)*N*N);
    REAL *B = malloc(sizeof(REAL)*N*N);
    REAL *C_parallel = malloc(sizeof(REAL)*N);
    REAL *C_base = malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i;
    int num_runs = 10;
    
    for (i=0; i<num_runs+1; i++)
        matvec_base(N, A, B, C_base);
        
    //warm up
    matvec_P3(N, A, B, C_parallel);

    elapsed_omp = read_timer();
    for (i=0; i<num_runs; i++)
        matvec_P3(N, A, B, C_parallel);
    elapsed_omp = (read_timer() - elapsed_omp)/num_runs;
    
    double error = check(C_base,C_parallel, N);


    printf("======================================================================================================\n");
    printf("\tMatrix Vector Addition: A[N][N] * B[N] = C[N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matvec_omp:\t\t%4f\t%4f\n", elapsed_omp * 1.0e3, ((((2.0) * N) * N) / (1.0e6 * elapsed_omp)));
    printf("error:%g\n", error);
    return 0;
}


