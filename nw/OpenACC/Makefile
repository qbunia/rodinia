# C compiler
CC = nvc++
CC_FLAGS = -g -O3 -acc

needle.out: needle.cpp
	$(CC) $(CC_FLAGS) needle.cpp -o needle.out

.PHONY: run clean

run: needle.out
	./needle.out 2048 10 2

clean:
	rm -f *.o *.out *.log
