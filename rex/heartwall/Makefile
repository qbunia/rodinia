ifeq (${LLVM_OPENMP_INSTALL},)
	LLVM_OPENMP_INSTALL := ${LLVM_PATH}
endif
ifeq (${CUDA_ARCH},)
	CUDA_ARCH := sm_70
endif

default: heartwall.out

heartwall.out: rex_lib_nvidia.cubin rose_main.c register_cubin.cpp ./AVI/avilib.o ./AVI/avimod.o
	clang -O3 -g rose_main.c register_cubin.cpp -I./AVI ./AVI/avilib.o ./AVI/avimod.o -o $@ -L${LLVM_OPENMP_INSTALL}/lib -lomptarget -lomp -lm

.PHONY:
# nvcc from CUDA 11.5 only officially supports GCC 11 or older. To use GCC 12, `-allow-unsupported-compiler` has to be specified.
rex_lib_nvidia.cubin: rex_lib_main.cu rex_nvidia.cu
	nvcc -allow-unsupported-compiler -O3 -rdc=true -cubin -arch ${CUDA_ARCH} -dlink $^ -o $@ -lm

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

.PHONY: run clean

run: heartwall.out
	./heartwall.out ../../data/heartwall/test.avi 20 4

# delete all object files
clean:
	rm -f *.o AVI/*.o *.out *.log *.cubin
