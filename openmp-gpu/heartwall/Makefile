ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CC = clang
CC_FLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64

# link objects(binaries) together
heartwall.out: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) $(CC_FLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall.out

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	$(CC) $(CC_FLAGS) $(OUTPUT) main.c -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o *.out *.log
