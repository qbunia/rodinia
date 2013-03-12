LOCAL_CC = gcc -g -O3 -Wall
CC = pathcc
CFLAGS = -lm -acc -Wall


all : nn

clean :
	rm -rf *.o nn

nn : nn_openacc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

