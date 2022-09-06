#include "backprop.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int layer_size = 0;

void backprop_face() {
  BPNN *net;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  // entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "usage: backprop <num of input elements>\n");
    exit(0);
  }

  layer_size = atoi(argv[1]);

  int seed;

  seed = 7;
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
