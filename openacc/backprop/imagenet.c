#include "backprop.h"
#include <stdio.h>
#include <stdlib.h>

extern int layer_size;

void load(BPNN *net) {
  float *units;
  int nr = layer_size;

  units = net->input_units;

  int k = 1;
  for (int i = 0; i < nr; i++) {
    units[k] = (float)rand() / RAND_MAX;
    k++;
  }
}
