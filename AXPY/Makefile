# Compiler and flags
COMPILER ?= clang
CC_CLANG := clang
CC_GCC := gcc
NVCC := nvcc
# Default optimization level is -O1
OPT_LEVEL := -O1
CFLAGS := -Wall -fopenmp $(OPT_LEVEL)
LDFLAGS :=

# Source files
SRCS := axpy_P0.c axpy_omp_CPU_P1.c axpy_omp_CPU_P2.c axpy_omp_CPU_P3.c main.c axpy_omp_Offloading.c axpy_common.c axpy_cuda.cu

# Object files
OBJS := $(SRCS:.c=.o) axpy_cuda.o

# Default executable name
DEFAULT_EXEC := axpy_program

# Default target
all: $(DEFAULT_EXEC)

# Linking the default executable
$(DEFAULT_EXEC): $(OBJS)
	@if [ "$(COMPILER)" = "clang" ]; then \
		$(CC_CLANG) $(CFLAGS) $(OBJS) -o $(DEFAULT_EXEC) $(LDFLAGS); \
	elif [ "$(COMPILER)" = "gcc" ]; then \
		$(CC_GCC) $(CFLAGS) $(OBJS) -o $(DEFAULT_EXEC); \
	else \
		echo "Invalid compiler choice. Use 'make COMPILER=clang' or 'make COMPILER=gcc'"; \
	fi

# Clean up
clean:
	rm -f $(filter-out %.cu, $(OBJS)) $(DEFAULT_EXEC) axpy_P0_exec axpy_omp_CPU_P1_exec axpy_omp_CPU_P2_exec axpy_omp_CPU_P3_exec axpy_omp_Offloading_exec axpy_cuda_exec

# Targets for different versions
axpy_P0_exec: axpy_P0.o main.o axpy_common.o
	$(CC_CLANG) $(CFLAGS) $^ -o $@ $(LDFLAGS)

axpy_omp_CPU_P1_exec: axpy_omp_CPU_P1.o main.o axpy_common.o
	$(CC_CLANG) $(CFLAGS) $^ -o $@ $(LDFLAGS)

axpy_omp_CPU_P2_exec: axpy_omp_CPU_P2.o main.o axpy_common.o
	$(CC_CLANG) $(CFLAGS) $^ -o $@ $(LDFLAGS)

axpy_omp_CPU_P3_exec: axpy_omp_CPU_P3.o main.o axpy_common.o
	$(CC_CLANG) $(CFLAGS) $^ -o $@ $(LDFLAGS)

axpy_omp_Offloading_exec: axpy_omp_Offloading.c main.c axpy_common.c
	$(CC_CLANG) $(CFLAGS) -fopenmp-targets=nvptx64 $^ -o $@ $(LDFLAGS)

axpy_cuda_exec: axpy_cuda.cu main.c axpy_common.c
	$(NVCC) -Xcompiler "$(CFLAGS)" $^ -o $@
