# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

C_C = clang
#OMP_LIB = -lgomp
OMP_FLAG = -O3 -fopenmp -fopenmp-targets=nvptx64

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu.o \
			./util/num/num.o \
			./util/timer/timer.o
	$(C_C)	main.o \
			./kernel/kernel_gpu.o \
			./util/num/num.o \
			./util/timer/timer.o \
			-lm \
			$(OMP_FLAG) \
			-o lavaMD
# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			./kernel/kernel_gpu.h \
			./kernel/kernel_gpu.c \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c
	$(C_C) main.c \
			-c \
			-o main.o \
			-O3

./kernel/kernel_gpu.o:	./kernel/kernel_gpu.h \
						./kernel/kernel_gpu.c
	$(C_C)				./kernel/kernel_gpu.c \
						-c \
						-o ./kernel/kernel_gpu.o \
						-O3 \
						$(OMP_FLAG)

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
                ./util/device/*.o \
		lavaMD
