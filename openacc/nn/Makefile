LOCAL_CC = gcc -g -O3 -Wall
CC = pathCC
CFLAGS = -O3 -acc


all : nn

clean :
	rm -rf *.o nn

nn : nn_openacc.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lm 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

