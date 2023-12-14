LLVM = clang++
LLVM_FLAGS = -fopenmp -fopenmp-targets=nvptx64

GCC = g++
GCC_FLAGS = -fopenmp

NV = nvc++
NV_FLAGS = -mp=gpu

CC_FLAGS = -g -O3 -Wall

sc_llvm.out: streamcluster.cpp
	$(LLVM) $(LLVM_FLAGS) $(CC_FLAGS) streamcluster.cpp -o $@

sc_gcc.out: streamcluster.cpp
	$(GCC) $(GCC_FLAGS) $(CC_FLAGS) streamcluster.cpp -o $@

sc_nv.out: streamcluster.cpp
	$(NV) $(NV_FLAGS) $(CC_FLAGS) streamcluster.cpp -o $@

all: sc_llvm.out sc_gcc.out sc_nv.out

.PHONY: run clean

run: sc_llvm.out
	./sc_llvm.out 10 20 256 65536 65536 1000 none output.log 1

clean:
	rm -f *.o *.out *.log
	
