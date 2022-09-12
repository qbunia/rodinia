CC = nvc
CC_FLAGS = -g -O3 -acc -Wall

# link objects(binaries) together
srad.out: main.o
	$(CC) $(CC_FLAGS) main.o -lm -o srad.out

# compile main function file into object (binary)
main.o: main.c define.c graphics.c
	$(CC) $(CC_FLAGS) main.c -c

.PHONY: run clean

run: srad.out
	./srad.out 100 0.5 502 458

# delete all object files
clean:
	rm -f *.o *.out *.pgm
