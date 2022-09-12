#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "timer.h"

void run(int argc, char **argv);

/* define timer macros */
#define pin_stats_reset() startCycle()
#define pin_stats_pause(cycles) stopCycle(cycles)
#define pin_stats_dump(cycles) printf("timer: %Lu\n", cycles)

#define BENCH_PRINT

#define NUM_TEAMS 256
#define NUM_THREADS 1024

int rows, cols;
int *data;
int *result;
#define M_SEED 9

void init(int argc, char **argv) {
  if (argc == 3) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
  } else {
    printf("Usage: pathfiner width num_of_steps\n");
    exit(0);
  }
  data = new int[rows * cols];
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      data[i * cols + j] = rand() % 10;
    }
  }
  for (int j = 0; j < cols; j++)
    result[j] = data[j];
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", data[i * cols + j]);
    }
    printf("\n");
  }
#endif
}

void fatal(char *s) { fprintf(stderr, "error: %s\n", s); }

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

int main(int argc, char **argv) {
  run(argc, argv);

  return EXIT_SUCCESS;
}

void run(int argc, char **argv) {
  init(argc, argv);

  unsigned long long cycles;

  int *src, *dst, *temp;
  int min;

  dst = result;
  src = new int[cols];

  pin_stats_reset();
#pragma omp target data map(tofrom                                             \
                            : src [0:cols], dst [0:cols])                      \
    map(to                                                                     \
        : data [0:rows * cols])
  {
    for (int t = 0; t < rows - 1; t++) {
      temp = src;
      src = dst;
      dst = temp;
#pragma omp target teams distribute parallel for private(min)                  \
    num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
      for (int n = 0; n < cols; n++) {
        min = src[n];
        if (n > 0)
          min = MIN(min, src[n - 1]);
        if (n < cols - 1)
          min = MIN(min, src[n + 1]);
        dst[n] = data[(t + 1) * cols + n] + min;
      }
    }
  }

  pin_stats_pause(cycles);
  pin_stats_dump(cycles);

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", dst[i]);
  printf("\n");
#endif

  delete[] data;
  delete[] dst;
  delete[] src;
}
