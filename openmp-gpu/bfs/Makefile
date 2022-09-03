CC = clang++
CC_FLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64

bfs.out: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs.out

run: bfs.out
	./bfs.out ../../data/bfs/graph1MW_6.txt

.PHONY: clean
clean:
	rm -f *.out *.o result.log
