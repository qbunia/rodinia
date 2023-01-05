ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif


# C compiler
CC = clang
CC_FLAGS = -g -O3 -Wall -fopenmp -lm -I$(MATRIX_DIR) -fopenmp-targets=nvptx64 -L/opt/llvm/llvm-14.x-install/lib 
 
# Matrix library path
MATRIX_DIR = ../meschach_lib


leukocyte: detect_main.o avilib.o find_ellipse.o track_ellipse.o misc_math.o $(MATRIX_DIR)/meschach.a
	$(CC) $(CC_FLAGS) -lm avilib.o find_ellipse.o track_ellipse.o misc_math.o detect_main.o -o leukocyte -lm $(MATRIX_DIR)/meschach.a -lpthread

avilib.o: avilib.c
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c
 
track_ellipse.o: track_ellipse.c
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c
 
misc_math.o: misc_math.c
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c

find_ellipse.o: find_ellipse.c
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

detect_main.o: find_ellipse.h track_ellipse.h avilib.h
find_ellipse.o: avilib.h
track_ellipse.o: find_ellipse.h track_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o leukocyte
