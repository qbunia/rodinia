include ../common/environment.mk

# Source files
SRCS := axpy_P0.c axpy_omp_CPU_P1.c axpy_omp_CPU_P2.c axpy_omp_CPU_P3.c main.c axpy_omp_Offloading.c axpy_common.c axpy_cuda.cu

# Object files
OBJS := $(SRCS:.c=.o) axpy_cuda.o

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS))  axpy_cuda_exec *exec *.csv

axpy_P0: axpy_P0.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o axpy_omp_CPU_P0_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_omp_CPU_P1: axpy_omp_CPU_P1.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o axpy_omp_CPU_P1_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_CPU_P2: axpy_omp_CPU_P2.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o axpy_omp_CPU_P2_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_CPU_P3: axpy_omp_CPU_P3.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o axpy_omp_CPU_P3_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_omp_Offloading_P1: axpy_omp_Offloading_P1.c main.c axpy_common.c
	$(CC) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o axpy_omp_Offloading_P1_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P2: axpy_omp_Offloading_P2.c main.c axpy_common.c
	$(CC) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o axpy_omp_Offloading_P2_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_omp_Offloading_P3: axpy_omp_Offloading_P3.c main.c axpy_common.c
	$(CC) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o axpy_omp_Offloading_P3_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P4: axpy_omp_Offloading_P4.c main.c axpy_common.c
	$(CC) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o axpy_omp_Offloading_P4_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)
	
axpy_cuda_exec: axpy_cuda.cu main.c axpy_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@
