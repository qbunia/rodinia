OMP_NUM_THREADS = 8
MIC_OMP_NUM_THREADS = 216

CC = clang++
CC_FLAGS = -g -O3 -fopenmp -fopenmp-targets=nvptx64

all: euler3d_cpu.out euler3d_cpu_double.out pre_euler3d_cpu pre_euler3d_cpu_double

euler3d_cpu.out: euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu.out

euler3d_cpu_double.out: euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double.out

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double

.PHONY: clean
clean:
	rm -f *.o *.out *.log
	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_cpu_double
