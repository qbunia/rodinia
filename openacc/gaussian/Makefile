include ../../common/make.config

CC := pathcc
CCFLAGS := -acc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.c
EXE = gaussian

release: $(SRC)
	$(CC) $(CCFLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
