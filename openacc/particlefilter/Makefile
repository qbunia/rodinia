#makefile

CC = pathcc
FLAGS = -acc

openmp: ex_particle_OPENACC_seq.c
	$(CC) -O3 -ffast-math $(FLAGS) ex_particle_OPENACC_seq.c -o particle_filter -lm
	
all: ex_particle_OPENACC_seq.c
	$(CC) $(FLAGS) ex_particle_OPENACC_seq.c -o particle_filter_psc -ffast-math -lm

clean:
	rm particle_filter
