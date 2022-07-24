CC = clang
CC_FLAGS = -g -fopenmp -O3 -lstdc++

all: bfs.out

bfs.out: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs.out

run: bfs.out
	./bfs.out ../../data/bfs/graph1MW_6.txt

clean:
	rm -f *.out *.o result.txt
