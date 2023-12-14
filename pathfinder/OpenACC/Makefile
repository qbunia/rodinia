GCC = gcc
GCC_FLAGS = -fopenacc

NV = nvc
NV_FLAGS = -acc

SRC = pathfinder.c
CC_FLAGS = -g -O3 -Wall

pf_nv.out: $(SRC)
	$(NV) $(NV_FLAGS) $(CC_FLAGS) $(SRC) -o $@

pf_gcc.out: $(SRC)
	$(GCC) $(GCC_FLAGS) $(CC_FLAGS) $(SRC) -o $@

all: pf_gcc.out pf_nv.out

.PHONY: run clean

run: pf_nv.out
	./pf_nv.out 100000 100 > result.log

clean:
	rm -f *.o *.out *.log
