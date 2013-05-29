# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
	
CC = pathcc
CC_FLAGS = -acc -g

# link objects(binaries) together
a.out:	main.o
	$(CC) $(CC_FLAGS) main.o -o srad -lm

# compile main function file into object (binary)
main.o: main.c define.c graphics.c
	$(CC) main.c -c -O3 $(CC_FLAGS)

# delete all object files
clean:
	rm *.o srad
