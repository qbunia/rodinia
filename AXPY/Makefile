CC = clang
CFLAGS = -fopenmp
OPT_LEVEL_P0 = -O0
OPT_LEVEL_P1 = -O2
OPT_LEVEL_P2 = -O3

all: axpy_omp_P0 axpy_omp_P1 axpy_omp_P2 axpy_omp_P3

axpy_omp_P0: axpy_omp_P0.c
	$(CC) $(OPT_LEVEL_P0) $< -o $@

axpy_omp_P1: axpy_omp_P1.c
	$(CC) $(CFLAGS) $(OPT_LEVEL_P1) $< -o $@

axpy_omp_P2: axpy_omp_P2.c
	$(CC) $(CFLAGS) $(OPT_LEVEL_P2) $< -o $@

axpy_omp_P3: axpy_omp_P3.c
	$(CC) $(CFLAGS) $(OPT_LEVEL_P2) $< -o $@

clean:
	rm -f axpy_omp_P0 axpy_omp_P1 axpy_omp_P2 axpy_omp_P3
