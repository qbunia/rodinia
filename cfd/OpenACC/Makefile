MIC_ACC_VECTOR_LENGTH = 216

CC = nvc++
CC_FLAGS = -g -O3 -acc

all: euler3d_cpu.out euler3d_cpu_double.out pre_euler3d_cpu.out pre_euler3d_cpu_double.out

euler3d_cpu.out: euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_ACC_VECTOR_LENGTH) euler3d_cpu.cpp -o euler3d_cpu.out

euler3d_cpu_double.out: euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_ACC_VECTOR_LENGTH) euler3d_cpu_double.cpp -o euler3d_cpu_double.out

pre_euler3d_cpu.out: pre_euler3d_cpu.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_ACC_VECTOR_LENGTH) pre_euler3d_cpu.cpp -o pre_euler3d_cpu.out

pre_euler3d_cpu_double.out: pre_euler3d_cpu_double.cpp
	$(CC) $(CC_FLAGS) -Dblock_length=$(MIC_ACC_VECTOR_LENGTH) pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double.out

.PHONY: run clean

run: euler3d_cpu.out
	./euler3d_cpu.out ../../data/cfd/fvcorr.domn.097K

clean:
	rm -f *.o *.out *.log
