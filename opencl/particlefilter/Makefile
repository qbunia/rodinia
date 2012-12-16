include ../../common/make.config

#OPENCL_DIR = /af21/den4gr/NVIDIA_GPU_Computing_SDK/

# C compiler
CC = g++
CC_FLAGS = -O2 
#INCLUDES = -I$(OPENCL_DIR)/OpenCL/common/inc -I$(OPENCL_DIR)/shared/inc/
INCLUDES = -I$(OPENCL_INC)

ALL: OCL_particlefilter_naive OCL_particlefilter_float

OCL_particlefilter_naive: ex_particle_OCL_naive_seq.o
	$(CC) $(CC_FLAGS) -lOpenCL ex_particle_OCL_naive_seq.o $(INCLUDES) -L$(OPENCL_LIB) -o OCL_particlefilter_naive 

OCL_particlefilter_float: ex_particle_OCL_float_seq.o
	$(CC) $(CC_FLAGS) -lOpenCL ex_particle_OCL_float_seq.o $(INCLUDES)  -L$(OPENCL_LIB) -o OCL_particlefilter_float 

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

ex_particle_OCL_naive_seq.o: ex_particle_OCL_naive_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_naive_seq.cpp -c

ex_particle_OCL_float_seq.o: ex_particle_OCL_float_seq.cpp
	$(CC) $(CC_FLAGS) $(INCLUDES) ex_particle_OCL_float_seq.cpp -c

clean:
	rm -f *.o *~ OCL_particlefilter_naive OCL_particlefilter_float *.linkinfo
