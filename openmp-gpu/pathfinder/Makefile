LLVM = clang
LLVM_FLAGS = -fopenmp -fopenmp-targets=nvptx64

NV = nvc
NV_FLAGS = -mp=gpu

GCC = gcc
GCC_FLAGS = -fopenmp

SRC = pathfinder.c
CC_FLAGS = -g -O3 -Wall

pf_llvm.out: $(SRC)
	$(LLVM) $(LLVM_FLAGS) $(CC_FLAGS) $(SRC) -o $@

pf_nv.out: $(SRC)
	$(NV) $(NV_FLAGS) $(CC_FLAGS) $(SRC) -o $@

pf_gcc.out: $(SRC)
	$(GCC) $(GCC_FLAGS) $(CC_FLAGS) $(SRC) -o $@

all: pf_llvm.out pf_gcc.out pf_nv.out

.PHONY: run clean

run: pf_llvm.out
	./pf_llvm.out 100000 100 > result.log

clean:
	rm -f *.o *.out *.log
