include ../../common/make.config

CC := pathcc
CCFLAGS := -acc

SRC = gaussian.c
EXE = gaussian

release: $(SRC)
	$(CC) $(CCFLAGS) $(SRC) -o $(EXE)

clean:
	rm gaussian
