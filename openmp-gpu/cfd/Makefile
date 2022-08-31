CUDA_SDK_PATH = /af10/sc5nf/NVIDIA_CUDA_SDK

OMP_NUM_THREADS = 8

CC = pathCC
CC_FLAGS = -acc -O3

all: euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double

euler3d_cpu: euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu
euler3d_cpu_double: euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double


clean:
	rm -f euler3d_cpu
	rm -f euler3d_cpu_double
	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_cpu_double
