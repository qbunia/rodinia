ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

CC_FLAGS = -Wall -g -O3 -lm

bt_llvm.out: main.c util/timer/timer.c util/num/num.c kernel/kernel_cpu.c kernel/kernel_cpu_2.c
	clang ${CC_FLAGS} -fopenmp -fopenmp-targets=nvptx64 -I. $^ -o $@ -L${LLVM_OPENMP_INSTALL}/lib

bt_gcc.out: main.c util/timer/timer.c util/num/num.c kernel/kernel_cpu.c kernel/kernel_cpu_2.c
	gcc ${CC_FLAGS} -fopenmp -I. $^ -o $@

bt_nvc.out: main.c util/timer/timer.c util/num/num.c kernel/kernel_cpu.c kernel/kernel_cpu_2.c
	nvc ${CC_FLAGS} -mp=gpu -I. $^ -o $@

all: bt_llvm.out bt_gcc.out bt_nvc.out

.PHONY: run clean

run: bt_llvm.out
	./$< core 2 file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt

clean:
	rm -f *.o *.out *.log *.cubin
