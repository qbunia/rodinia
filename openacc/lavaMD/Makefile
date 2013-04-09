# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

#C_C = pathcc
#C_C_FLAG = -acc
C_C = pgcc
C_C_FLAG = -acc -Minfo -I /usr/include/i386-linux-gnu/ -L /usr/lib/i386-linux-gnu/ -g

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_acc.o \
			./util/num/num.o \
			./util/timer/timer.o
	$(C_C) $(C_C_FLAG) main.o \
			./kernel/kernel_acc.o \
			./util/num/num.o \
			./util/timer/timer.o \
			-lm \
			-o lavaMD
# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_acc.h \
			./kernel/kernel_acc.c \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c
	$(C_C) $(C_C_FLAG) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_acc.o:	./kernel/kernel_acc.h \
						./kernel/kernel_acc.c
	$(C_C) $(C_C_FLAG) 	./kernel/kernel_acc.c \
						-c \
						-o ./kernel/kernel_acc.o \
						-O3

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C) $(C_C_FLAG) ./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C) $(C_C_FLAG) 	./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		lavaMD
