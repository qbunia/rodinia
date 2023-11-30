# tests for the Microbenchmarks
#Question: I did hard coding for the compile and execution, since there is different optimization file for different files for one application and different instructions for the different application. Any smarter methods?

mkdir ResultsForMicroBenchmarks
cd ResultsForMicroBenchmarks
mkdir Results_AXPY
cd ..

mkdir ResultsForRodiniaBenchmarks
cd ResultsForRodiniaBenchmarks
mkdir backprop cfd hotspot3D leukocyte mummergpu nw srad bfs heartwall kmeans lud myocyte particlefilter streamcluster b+tree hotspot lavaMD MicroBenchmarks nn pathfinder
cd ..


cd openmp/MicroBenchmarks

#test for AXPY, input size: 102400000
cd AXPY
clang axpy_omp_P0.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompCPU_P0_result.txt

clang -fopenmp -O2 axpy_omp_P1.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompCPU_P1_result.txt

clang -fopenmp -O3 axpy_omp_P2.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompCPU_P2_result.txt

clang -fopenmp -O3 axpy_omp_P3.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompCPU_P3_result.txt

rm test


cd ../../../openmp-gpu/MicroBenchmarks/AXPY
clang -fopenmp -O2 -fopenmp-targets=nvptx64 axpy_omp_Offloading_P1.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompGPU_P1_result.txt

clang -fopenmp -O2 -fopenmp-targets=nvptx64 axpy_omp_Offloading_P2.c -o test
./test 102400000 -> ../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_ompGPU_P2_result.txt

rm test


cd ../../../cuda/MicroBenchmarks/AXPY
cd P1
nvcc -o test axpy_cuda.c axpy_cudakernel.cu
./test 102400000 -> ../../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_cuda_P1_result.txt
rm test

cd ../P2
nvcc -o test axpy_cuda.c axpy_cudakernel.cu
./test 102400000 -> ../../../../ResultsForMicroBenchmarks/Results_AXPY/AXPY_cuda_P2_result.txt
rm test



#test for other benchmarks

#b++ tree

cd openmp/b+tree
make
bash run -> ../../ResultsForRodiniaBenchmarks/b+tree/OMP_CPU_B+Tree.txt
make clean

cd ../../openmp-gpu/b+tree
make
bash run -> ../../ResultsForRodiniaBenchmarks/b+tree/OMP_GPU_B+Tree.txt
make clean

cd ../../cuda/b+tree
make
bash run -> ../../ResultsForRodiniaBenchmarks/b+tree/cuda_B+Tree.txt
make clean
