ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CC = nvc
CC_FLAGS = -O3 -acc

# link objects(binaries) together
heartwall.out: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) $(CC_FLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall.out

# compile main function file into object (binary)
main.o: main.c define.h kernel.c
	$(CC) $(CC_FLAGS) $(OUTPUT) main.c -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

.PHONY: run clean

run: heartwall.out
	./heartwall.out ../../data/heartwall/test.avi 20 4

# delete all object files
clean:
	rm -f *.o AVI/*.o *.out *.log
