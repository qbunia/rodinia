CC = nvc++
SRC = pathfinder.cpp
EXE = pathfinder.out
FLAGS = -O3 -acc

$(EXE): $(SRC)
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug: $(SRC)
	$(CC) $(SRC) -g -Wall -o $(EXE)

.PHONY: run clean

run: $(EXE)
	./pathfinder.out 100000 100 > result.log

clean:
	rm -f *.o *.out *.log
