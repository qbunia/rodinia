#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define REAL float
#define BLOCK_SIZE 1024

int main(int argc, char** argv) {

    int size = 1 << 24;    // number of elements to reduce

    if (argc == 2) {
        size = atoi(argv[1]);
    };

    printf("Total number of elements: %d\n", size);


    // create random input data on CPU
    unsigned int bytes = size * sizeof(REAL);

    REAL *h_idata = (REAL *)malloc(bytes);

    for (int i = 0; i < size; i++) {
      // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (rand() & 0xFF) / (REAL)RAND_MAX;
    }

    double sum = 0;
#pragma omp target teams distribute parallel for map(to: h_idata[0:size]) map(from: sum) num_teams(size/BLOCK_SIZE) num_threads(BLOCK_SIZE) reduction(+: sum)
    for (int i = 0; i < size; i++)
        sum += h_idata[i];

    printf("Sum: %lg\n", sum);

    return 0;
}
