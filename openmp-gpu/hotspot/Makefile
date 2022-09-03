# C++ compiler
CC = clang++
CC_FLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64

hotspot.out: hotspot.cpp
	$(CC) $(CC_FLAGS) hotspot.cpp -o hotspot.out

run: hotspot.out
	./hotspot.out 512 512 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512

.PHONY: clean
clean:
	rm -f *.o *.out
