ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

pf.out: rex_lib_nvidia.cubin rose_pathfinder.c register_cubin.cpp
	clang -O3 -g rose_pathfinder.c register_cubin.cpp -o $@ -L${LLVM_OPENMP_INSTALL}/lib -lomptarget -lomp

.PHONY:
# nvcc from CUDA 11.5 only officially supports GCC 11 or older. To use GCC 12, `-allow-unsupported-compiler` has to be specified.
rex_lib_nvidia.cubin: rex_lib_pathfinder.cu rex_nvidia.cu
	nvcc -allow-unsupported-compiler -O3 -rdc=true -cubin -arch ${CUDA_ARCH} -dlink $^ -o $@

.PHONY: run clean

run: pf.out
	./$< 100000 1000 > result.log

clean:
	rm -f *.o *.out *.log *.cubin
