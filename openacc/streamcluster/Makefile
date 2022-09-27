CC = nvc++
CC_FLAGS = -g -O3 -acc -Wall

sc.out: streamcluster.cpp
	$(CC) $(CC_FLAGS) streamcluster.cpp -o sc.out

.PHONY: run clean

run: sc.out
	./sc.out 10 20 256 65536 65536 1000 none output.log 1

clean:
	rm -f *.o *.out *.log
	
