C_C = clang
#OMP_LIB = -lgomp
OMP_FLAG = -O3 -g -Wall -fopenmp -lm -fopenmp-targets=nvptx64


SRC = 3D.c

EXE = 3D 

OUTPUT = *.out

release: $(SRC)
	$(C_C) $(OMP_FLAG) $(SRC) -o $(EXE)

enum: $(SRC)
	$(C_C) $(OMP_FLAG)  $(SRC) -o $(EXE) 

debug: $(SRC)
	$(C_C) $(OMP_FLAG) -g $(SRC) -o $(EXE) 

debugenum: $(SRC)
	$(C_C) $(OMP_FLAG) -g  $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 

