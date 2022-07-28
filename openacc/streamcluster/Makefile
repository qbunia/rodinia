include ../../common/make.config

CC = pathcc
CC_FLAGS = -g -O3 -acc

TARGET_G = sc_gpu

$(TARGET_G): streamcluster.cpp streamcluster_kernel.cpp
	$(CC) $(CC_FLAGS) streamcluster.cpp streamcluster_kernel.cpp -o $(TARGET_G)
	

clean:
	rm -f *.o *~ *.txt $(TARGET_G) *.linkinfo
	
