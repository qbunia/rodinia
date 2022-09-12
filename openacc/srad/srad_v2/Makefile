# C compiler
CC = nvc++
CC_FLAGS = -g -O3 -acc -Wall

srad.out: srad.cpp
	$(CC) $(CC_FLAGS) srad.cpp -o srad.out 

.PHONY: run clean

run: srad.out
	./srad.out 2048 2048 0 127 0 127 0.5 2

clean:
	rm -f *.o *.out
