# C compiler
CC = clang++
CC_FLAGS = -g -O3 -fopenmp -fopenmp-targets=nvptx64

needle.out: needle.cpp
	$(CC) $(CC_FLAGS) needle.cpp -o needle.out

.PHONY: run clean

run: needle.out
	./needle.out 2048 10 2

clean:
	rm -f *.o *.out *.log
