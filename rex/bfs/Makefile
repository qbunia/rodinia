ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

default: bfs.out

all: bfs.out

.PHONY:
clean: 
	rm -rf ${OBJS} *.log *.out *.cubin

bfs.out: rex_lib_nvidia.cubin rose_bfs.cpp register_cubin.cpp
	clang -O3 -g rose_bfs.cpp register_cubin.cpp -o $@ -L${LLVM_OPENMP_INSTALL}/lib -lomptarget -lomp

.PHONY:
# nvcc from CUDA 11.5 only officially supports GCC 11 or older. To use GCC 12, `-allow-unsupported-compiler` has to be specified.
rex_lib_nvidia.cubin: rex_lib_bfs.cu rex_nvidia.cu
	nvcc -allow-unsupported-compiler -O3 -rdc=true -cubin -arch ${CUDA_ARCH} -dlink rex_lib_bfs.cu rex_nvidia.cu -o rex_lib_nvidia.cubin

