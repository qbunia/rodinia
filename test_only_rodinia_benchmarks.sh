#!/bin/bash

# Function to compile and run benchmarks
run_benchmark() {
    cd "$1"
    make
    bash run -> "../../ResultsForRodiniaBenchmarks/$2/$3.txt"
    make clean
    cd ../../
}

# B+ Tree benchmarks
run_benchmark "openmp/b+tree" "b+tree" "OMP_CPU_B+Tree"
run_benchmark "openmp-gpu/b+tree" "b+tree" "OMP_GPU_B+Tree"
run_benchmark "cuda/b+tree" "b+tree" "cuda_B+Tree"

# Backprop benchmarks
run_benchmark "openmp/backprop" "backprop" "OMP_CPU_backprop"
run_benchmark "openmp-gpu/backprop" "backprop" "OMP_GPU_backprop"
run_benchmark "cuda/backprop" "backprop" "cuda_backprop"


