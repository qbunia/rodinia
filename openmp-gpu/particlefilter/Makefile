#makefile

CC = pgcc
FLAGS = -acc -Minfo

openmp: ex_particle_OPENACC_seq.c
	$(CC) -O3 $(FLAGS) ex_particle_OPENACC_seq.c -o particle_filter -lm
	
all: ex_particle_OPENACC_seq.c
	$(CC) $(FLAGS) ex_particle_OPENACC_seq.c -o particle_filter_psc -lm

clean:
	rm particle_filter
