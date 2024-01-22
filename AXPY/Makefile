include ../common/environment.mk

# Source files
SRCS := axpy_P0.c axpy_omp_CPU_P1.c axpy_omp_CPU_P2.c axpy_omp_CPU_P3.c main.c axpy_omp_Offloading.c axpy_common.c axpy_cuda.cu

# Object files
OBJS := $(SRCS:.c=.o) axpy_cuda.o

# Default executable name
DEFAULT_EXEC := axpy_program

# Default target
all: $(DEFAULT_EXEC)


# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) $(DEFAULT_EXEC) axpy_cuda_exec *exec *.csv

axpy_P0_exec: axpy_P0.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o axpy_omp_CPU_P0_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS) $(LDFLAGS)
	
axpy_omp_CPU_P%: axpy_omp_CPU_P%.o main.o axpy_common.o
	$(CC) $(CFLAGS) $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_omp_Offloading_P%: axpy_omp_Offloading_P%.c main.c axpy_common.c
	$(CC) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o $@_$(CC)_$(OPT_LEVEL)_exec $(LDFLAGS)

axpy_cuda_exec: axpy_cuda.cu main.c axpy_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@
