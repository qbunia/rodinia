include ../../common/make.config

#C compiler
CC = g++

SRC = bfs.cpp timer.cc

CC_FLAGS = -g -O3

EXE = bfs

release:$(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)

errmsg:$(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D ERRMSG PTX_MSG
	
ptx:$(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D PROFILING
	
res:$(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -D RES_MSG
	
debug: $(SRC)
	$(CC) $(CC_FLAGS) -lOpenCL $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)
	
run:
	./$(EXE)
	
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
