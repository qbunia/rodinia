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
#include <omp.h>
#define TEAM_NUM 1024
#define TEAM_SIZE 256


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

REAL check(REAL*A, REAL*B, int n)
{
    int i;
    REAL diffsum =0.0, sum = 0.0;
    for (i = 0; i < n; i++) {
        diffsum += fabs(A[i] - B[i]);
        sum += fabs(B[i]);
    }
    return diffsum;
}

void matvec_base(int N, REAL *A, REAL *B, REAL *C) {
    int i,j;
    REAL temp;
    int size = N*N;
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
    }
}

void matvec_omp_offloading_P1(int N, REAL *A, REAL *B, REAL *C) {
    int i,j;
    REAL temp;
    int size = N*N;
    #pragma omp target teams distribute parallel for map(to: N, A[0:size], B[0:N]) map(from: C[0:N]) num_teams(TEAM_NUM) num_threads(TEAM_SIZE)
    for (i = 0; i < N; i++) {
        temp = 0.0;
        for (j = 0; j < N; j++)
            temp += A[i * N + j] * B[j];
        C[i] = temp;
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
    REAL *C_omp = malloc(sizeof(REAL)*N);
    REAL *C_serial = malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i;
    int num_runs = 10;
    
    for (i=0; i<num_runs+1; i++)
        matvec_base(N, A, B, C_serial);

    //warm up
    matvec_omp_offloading_P1(N, A, B, C_omp);
    
    elapsed_omp = read_timer();
    for (i=0; i<num_runs; i++)
        matvec_omp_offloading_P1(N, A, B, C_omp);
    elapsed_omp = (read_timer() - elapsed_omp)/num_runs;


    printf("======================================================================================================\n");
    printf("\tMatrix Vector Addition: A[N][N] * B[N] = C[N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matvec_omp_offloading_P1:\t\t%4f\t%4f\n", elapsed_omp * 1.0e3, ((((2.0) * N) * N) / (1.0e6 * elapsed_omp)));
    printf("Check:%f\n",check(C_serial,C_omp,N));
    return 0;
}


