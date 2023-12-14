CC = clang
CC_FLAGS = -O3 -lm -fopenmp -fopenmp-targets=nvptx64 -Wall

nn.out: nn.c
	$(CC) -o $@ $< $(CC_FLAGS)

hurricane_gen: hurricane_gen.c
	$(CC) -o $@ $< -lm

.PHONY: run clean

run: nn.out
	./nn.out filelist_4 5 30 90

clean:
	rm -rf *.o *.out
