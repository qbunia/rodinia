#makefile

openmp: ex_particle_OPENMP_seq.c
	gcc -lm -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter


icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


clean:
	rm particle_filter
