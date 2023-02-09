#include "rex_kmp.h"
#include <stdio.h>
#include <stdlib.h>

// clang++ -g -c register_cubin.cpp -o register_cubin.o

#ifdef __cplusplus
extern "C" {
#endif

struct __tgt_bin_desc *__cubin_desc = 0;

void __attribute__((destructor)) unregister_kernel_entries() {
  __tgt_unregister_lib(__cubin_desc);
}

extern struct __tgt_offload_entry __start_omp_offloading_entries;
extern struct __tgt_offload_entry __stop_omp_offloading_entries;

struct __tgt_bin_desc *register_cubin(const char *filename) {

  // read cuda object file to char array
  FILE *file = fopen(filename, "r+");
  if (file == NULL) {
    return NULL;
  };
  fseek(file, 0, SEEK_END);
  long int size = ftell(file);
  fclose(file);
  // Reading data to array of unsigned chars
  file = fopen(filename, "r+");
  unsigned char *image = (unsigned char *)malloc(size);
  int bytes_read = fread(image, sizeof(unsigned char), size, file);
  fclose(file);

  /* init struct __tgt_device_image */
  struct __tgt_device_image *device_image =
      (struct __tgt_device_image *)malloc(sizeof(struct __tgt_device_image));
  device_image->ImageStart = image;
  device_image->ImageEnd = image + size;
  device_image->EntriesBegin = &__start_omp_offloading_entries;
  device_image->EntriesEnd = &__stop_omp_offloading_entries;

  struct __tgt_bin_desc *bin_desc =
      (struct __tgt_bin_desc *)malloc(sizeof(struct __tgt_bin_desc));

  bin_desc->NumDeviceImages = 1;
  bin_desc->DeviceImages = device_image;
  bin_desc->HostEntriesBegin = &__start_omp_offloading_entries;
  bin_desc->HostEntriesEnd = &__stop_omp_offloading_entries;

  __tgt_register_lib(bin_desc);
  return bin_desc;
}

void __attribute__((constructor)) register_kernel_entries() {
  char cuda_entry_name[] = "rex_lib_nvidia.cubin";
  __cubin_desc = register_cubin(cuda_entry_name);
}

#ifdef __cplusplus
}
#endif
