C_C = pgcc
#OMP_LIB = -lgomp
#OMP_FLAG = -O3 -g -Wall -fopenmp -lm -fopenmp-targets=nvptx64
ACC_FLAG = -g -O3 -acc -Wall

SRC = 3D.c

EXE = 3D 

OUTPUT = *.out

release: $(SRC)
	$(C_C) $(ACC_FLAG) $(SRC) -o $(EXE)

enum: $(SRC)
	$(C_C) $(ACC_FLAG)  $(SRC) -o $(EXE) 

debug: $(SRC)
	$(C_C) $(ACC_FLAG) -g $(SRC) -o $(EXE) 

debugenum: $(SRC)
	$(C_C) $(ACC_FLAG) -g  $(SRC) -o $(EXE) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 

