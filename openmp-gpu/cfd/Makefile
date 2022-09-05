OMP_NUM_THREADS = 8
MIC_OMP_NUM_THREADS = 216

all: euler3d_cpu_offload.out euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double

euler3d_cpu_offload.out: euler3d_cpu.cpp
	clang++ -g -O3 -Dblock_length=$(MIC_OMP_NUM_THREADS) -fopenmp -fopenmp-targets=nvptx64 euler3d_cpu.cpp -o euler3d_cpu_offload.out

euler3d_cpu_double: euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu_double.cpp -o euler3d_cpu_double

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double

.PHONY: clean
clean:
	rm -f *.o *.out *.log
	rm -f euler3d_cpu_double
	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_cpu_double
