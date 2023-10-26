#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/timeb.h>

#define REAL double
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define TEST 10
#define PROBLEM 10240
#define TEAM_NUM 1024
#define TEAM_SIZE 256
// clang -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_35 --cuda-path=/usr/local/cuda -O3 -lpthread -fpermissive -msse4.1 stencil_metadirective.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void stencil_omp(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);

static double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

void print_array(char *title, char *name, REAL *A, int n, int m) {
    printf("%s:\n", title);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%s[%d][%d]:%f  ", name, i, j, A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialize(int width, int height, REAL *u) {
    int i;
    int N = width*height;

    for (i = 0; i < N; i++)
        u[i] = rand() % 256;
}

int main(int argc, char *argv[]) {
    int n = PROBLEM;
    int m = PROBLEM;

    if (argc > 2) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    };

    REAL *input = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_cpu = (REAL *) malloc(sizeof(REAL) * n * m);
    initialize(n, m, input);

    const float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0,  0, 1, 0, 0, },
        { 0,  0, 2, 0, 0, },
        { 3,  4, 5, 6, 7, },
        { 0,  0, 8, 0, 0, },
        { 0,  0, 9, 0, 0, },
    };

    int width = m;
    int height = n;

    double elapsed = read_timer_ms();
    double cpu_time = 0.0;
    int i;

    for (i = 0; i < TEST; i++) {
        elapsed = read_timer_ms();
        stencil_omp(input, result_cpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        cpu_time += read_timer_ms() - elapsed;
    };
    printf("Problem Size: %d, %d\n", m,n);
    printf("CPU time(ms): %g\n", cpu_time/TEST);
    printf("CPU total time(ms): %g\n", cpu_time);

    double dif = 0;
    for (i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += fabs(result[i] - result_cpu[i]);
    }
    printf("verify dif = %g\n", dif);
    free(input);
    free(result);
    return 0;
}

void stencil_omp(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int i, j;
#pragma omp parallel for
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            REAL sum = 0;
            int m, n;
            for (n = 0; n < flt_width; n++) {
                for (m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}
