# C compiler
CC = clang
CC_FLAGS = -g -O3 -fopenmp -fopenmp-targets=nvptx64

backprop.out: backprop.o facetrain.o imagenet.o backprop_kernel.o
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o backprop.out -lm

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

.PHONY: run clean

run: backprop.out
	./backprop.out 65536

clean:
	rm -f *.o *.out
