CC = nvc++
CC_FLAGS = -O3 -acc

bfs.out: bfs.cpp
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs.out

.PHONY: run clean

run: bfs.out
	./bfs.out ../../data/bfs/graph1MW_6.txt

clean:
	rm -f *.out *.o result.log
