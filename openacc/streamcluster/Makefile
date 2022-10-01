GCC = g++
GCC_FLAGS = -fopenacc

NV = nvc++
NV_FLAGS = -acc

CC_FLAGS = -g -O3 -Wall

sc_nv.out: streamcluster.cpp
	$(NV) $(NV_FLAGS) $(CC_FLAGS) streamcluster.cpp -o $@

sc_gcc.out: streamcluster.cpp
	$(GCC) $(GCC_FLAGS) $(CC_FLAGS) streamcluster.cpp -o $@

all: sc_gcc.out sc_nv.out

.PHONY: run clean

run: sc_nv.out
	./sc_nv.out 10 20 256 65536 65536 1000 none output.log 1

clean:
	rm -f *.o *.out *.log
	
