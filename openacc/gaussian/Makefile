include ../../common/make.config

CC := nvc
CCFLAGS := -O3 -acc

SRC = gaussian.c
EXE = gaussian.out

$(EXE): $(SRC)
	$(CC) $(CCFLAGS) $(SRC) -o $(EXE)

run: $(EXE)
	./$(EXE) ../../data/gaussian/matrix4.txt

.PHONY: clean
clean:
	rm -f *.o *.out
