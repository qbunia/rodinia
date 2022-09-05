MIC_OMP_NUM_THREADS = 216

CC = clang++
CC_FLAGS = -g -O3 -fopenmp -fopenmp-targets=nvptx64

all: euler3d_cpu.out euler3d_cpu_double.out pre_euler3d_cpu.out pre_euler3d_cpu_double.out

euler3d_cpu.out: euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu.out

euler3d_cpu_double.out: euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double.out

pre_euler3d_cpu.out: pre_euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu.out

pre_euler3d_cpu_double.out: pre_euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double.out

.PHONY: clean
clean:
	rm -f *.o *.out *.log
