# C compiler
CC = nvc
CC_FLAGS = -g -O3 -acc

kmeans.out: cluster.o getopt.o kmeans.o kmeans_clustering.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans.out

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c
	$(CC) $(CC_FLAGS) cluster.c -c

getopt.o: getopt.c
	$(CC) $(CC_FLAGS) getopt.c -c

kmeans.o: kmeans.c
	$(CC) $(CC_FLAGS) kmeans.c -c

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

.PHONY: run clean

run: kmeans.out
	./kmeans.out -n 4 -i ../../data/kmeans/kdd_cup

clean:
	rm -f *.o *.out
