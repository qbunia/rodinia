# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

CC = pathcc
CCFLAGS = -acc

# link objects(binaries) together
myocyte.out:	main.o
	$(CC)	main.o \
			-lm $(CCFLAGS) \
	-o myocyte.out

# compile main function file into object (binary)
main.o: 	main.c \
				define.c \
				ecc.c \
				cam.c \
				fin.c \
				master.c \
				embedded_fehlberg_7_8.c \
				solver.c \
				file.c \
				timer.c
	$(CC)	main.c \
			-c -O3 $(CCFLAGS)

# delete all object files
clean:
	rm *.o myocyte.out output.txt
