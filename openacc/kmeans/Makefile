include ../../common/make.config

# C compiler
CC = pathcc
CC_FLAGS = -g -acc -O2 

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_kernel.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_kernel.o: kmeans_kernel.c 
	$(CC) $(CC_FLAGS) -c kmeans_kernel.c

clean:
	rm -f *.o *~ kmeans
