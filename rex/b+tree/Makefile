ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

b+tree: b+tree.out rex_lib_nvidia.cubin

b+tree.out: rose_main.c register_cubin.cpp timer.o num.o kernel/rose_kernel_cpu.c kernel/rose_kernel_cpu_2.c
	clang -O3 -g -I. $^ -o $@ -L${LLVM_OPENMP_INSTALL}/lib -lomptarget -lomp -lm

timer.o: util/timer/timer.c
	clang -O3 -g -c $^ -o $@	

num.o: util/num/num.c
	clang -O3 -g -c $^ -o $@	

.PHONY:
# nvcc from CUDA 11.5 only officially supports GCC 11 or older. To use GCC 12, `-allow-unsupported-compiler` has to be specified.
rex_lib_nvidia.cubin: kernel/rex_lib_kernel_cpu.cu kernel/rex_lib_kernel_cpu_2.cu rex_nvidia.cu
	nvcc -allow-unsupported-compiler -O3 -rdc=true -cubin -arch ${CUDA_ARCH} -dlink -I. $^ -o $@

.PHONY: run clean

run: b+tree
	./b+tree.out core 2 file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt

clean:
	rm -f *.o *.out *.log *.cubin
