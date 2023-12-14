PREFIX=${PARSECDIR}/pkgs/kernels/streamcluster/inst/${PARSECPLAT}

TARGET_C = sc_cpu.out

TARGET_O = sc_omp.out

ifdef version
  ifeq "$(version)" "parallel"
    CXXFLAGS :=	$(CXXFLAGS) -DENABLE_THREADS -pthread
  endif
endif

all: cpu omp

cpu: $(TARGET_C)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) streamcluster_original.cpp -o $(TARGET_C) $(LIBS) -DENABLE_THREADS -pthread

omp: $(TARGET_O)
	clang++ -Wall -g -O3 -fopenmp -o $(TARGET_O) streamcluster_omp.cpp

.PHONY: clean
clean:
	rm -f *.o *.out *.log
