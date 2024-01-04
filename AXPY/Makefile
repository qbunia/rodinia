# Compiler and flags
COMPILER ?= clang
CC_CLANG := clang
CC_GCC := gcc
# Default optimization level is -O1
OPT_LEVEL := -O1
CFLAGS := -Wall -fopenmp $(OPT_LEVEL)
LDFLAGS := -fopenmp-targets=nvptx64

# Source files
SRCS := axpy_omp_CPU_P1.c axpy_omp_CPU_P2.c axpy_omp_CPU_P3.c axpy_P0.c main.c axpy_omp_Offloading.c

# Object files
OBJS := $(SRCS:.c=.o)

# Executable name
EXEC := axpy_program

# Default target
all: $(EXEC)

# Linking the executable
$(EXEC): $(OBJS)
	@if [ "$(COMPILER)" = "clang" ]; then \
		$(CC_CLANG) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS); \
	elif [ "$(COMPILER)" = "gcc" ]; then \
		$(CC_GCC) $(CFLAGS) $(OBJS) -o $(EXEC); \
	else \
		echo "Invalid compiler choice. Use 'make COMPILER=clang' or 'make COMPILER=gcc'"; \
	fi

# Compiling individual source files
%.o: %.c
	# Use specific flags for axpy_omp_Offloading.c
	if [ "$<" = "axpy_omp_Offloading.c" ]; then \
		if [ "$(COMPILER)" = "clang" ]; then \
			$(CC_CLANG) -fopenmp -O2 -fopenmp-targets=nvptx64 -c $< -o $@; \
		elif [ "$(COMPILER)" = "gcc" ]; then \
			$(CC_GCC) -fopenmp -O2 -c $< -o $@; \
		fi \
	else \
		if [ "$(COMPILER)" = "clang" ]; then \
			$(CC_CLANG) $(CFLAGS) -c $< -o $@; \
		elif [ "$(COMPILER)" = "gcc" ]; then \
			$(CC_GCC) $(CFLAGS) -c $< -o $@; \
		fi \
	fi

# Clean up
clean:
	rm -f $(OBJS) $(EXEC)
