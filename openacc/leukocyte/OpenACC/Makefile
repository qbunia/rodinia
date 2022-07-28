include ../../../common/make.config

# C compiler
CC = pathcc
CC_FLAGS = -g -O3 -Wall -acc -I$(MATRIX_DIR)

# Matrix library path
MATRIX_DIR = ../meschach_lib


leukocyte: detect_main.o avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o $(MATRIX_DIR)/meschach.a
	$(CC) $(CC_FLAGS) -lm avilib.o find_ellipse.o find_ellipse_kernel.o track_ellipse.o track_ellipse_kernel.o misc_math.o detect_main.o -o leukocyte $(MATRIX_DIR)/meschach.a -L$(CUDA_LIB_DIR) -lm -lcuda -lcudart

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

detect_main.o: detect_main.c find_ellipse.h track_ellipse.h avilib.h 
	$(CC) $(CC_FLAGS) detect_main.c -c

find_ellipse_kernel.o: find_ellipse_kernel.c find_ellipse_kernel.h
	$(CC) $(CC_FLAGS) -c find_ellipse_kernel.c

track_ellipse_kernel.o: track_ellipse_kernel.c track_ellipse_kernel.h
	$(CC) $(CC_FLAGS) -I$(MATRIX_DIR) -c track_ellipse_kernel.c

find_ellipse.o: avilib.h

track_ellipse.o: find_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ leukocyte *.linkinfo
