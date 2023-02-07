ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

default: srad.out

all: srad.out

.PHONY:
clean: 
	rm -rf *.pgm *.o *.log *.out *.cubin

srad.out: rex_lib_nvidia.cubin rose_main.c register_cubin.cpp define.c graphics.c
	clang -O3 -g rose_main.c register_cubin.cpp -o $@ -L${LLVM_OPENMP_INSTALL}/lib -lomptarget -lm

run: srad.out
	./srad.out 100 0.5 502 458

.PHONY:
# nvcc from CUDA 11.5 only officially supports GCC 11 or older. To use GCC 12, `-allow-unsupported-compiler` has to be specified.
rex_lib_nvidia.cubin: rex_lib_main.cu rex_nvidia.cu
	nvcc -allow-unsupported-compiler -O3 -rdc=true -cubin -arch ${CUDA_ARCH} -dlink $^ -o $@

