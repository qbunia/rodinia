#include <stddef.h>
#include <stdint.h>

typedef struct ident {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char const *psource;
} ident_t;

struct __tgt_offload_entry {
  void *addr;       // Pointer to the offload entry info (function or global)
  char *name;       // Name of the function or global
  size_t size;      // Size of the entry info (0 if it is a function)
  int32_t flags;    // Flags associated with the entry, e.g. 'link'.
  int32_t reserved; // Reserved, to be used by the runtime library.
};

struct __tgt_device_image {
  void *ImageStart; // Pointer to the target code start
  void *ImageEnd;   // Pointer to the target code end
  struct __tgt_offload_entry
      *EntriesBegin; // Begin of table with all target entries
  struct __tgt_offload_entry *EntriesEnd; // End of table (non inclusive)
};

struct __tgt_bin_desc {
  int32_t NumDeviceImages; // Number of device types supported
  struct __tgt_device_image
      *DeviceImages; // Array of device images (1 per dev. type)
  struct __tgt_offload_entry
      *HostEntriesBegin; // Begin of table with all host entries
  struct __tgt_offload_entry *HostEntriesEnd; // End of table (non inclusive)
};

#ifdef __cplusplus
extern "C" {
#endif

void __kmpc_fork_call(ident_t *, int, void *, ...);
void __kmpc_atomic_start(void);
void __kmpc_atomic_end(void);
void __kmpc_push_num_threads(ident_t *, int, int);
int __kmpc_global_thread_num(ident_t *);
int __kmpc_single(ident_t *, int);
void __kmpc_end_single(ident_t *, int);
void __kmpc_barrier(ident_t *, int);
int __kmpc_serialized_parallel(ident_t *, int);
void __kmpc_end_serialized_parallel(ident_t *, int);
void __kmpc_for_static_init_4(ident_t *, int, int, int *, int *, int *, int *,
                              int, int);
void __kmpc_for_static_fini(ident_t *, int);
void __kmpc_dispatch_init_4(ident_t *, int, int, int, int, int, int);
int __kmpc_dispatch_next_4(ident_t *, int, int *, int *, int *, int *);

int __tgt_target_teams(int64_t device_id, void *host_ptr, int32_t arg_num,
                       void **args_base, void **args, int64_t *arg_sizes,
                       int64_t *arg_types, int32_t num_teams,
                       int32_t thread_limit);

// creates the host to target data mapping, stores it in the
// libomptarget.so internal structure (an entry in a stack of data maps) and
// passes the data to the device;
void __tgt_target_data_begin(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                             void **Args, int64_t *ArgSizes, int64_t *ArgTypes);

// passes data from the target, release target memory and destroys the
// host-target mapping (top entry from the stack of data maps) created by
// the last __tgt_target_data_begin
void __tgt_target_data_end(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                           void **Args, int64_t *ArgSizes, int64_t *ArgTypes);

void __tgt_target_data_update(int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
                              void **Args, int64_t *ArgSizes,
                              int64_t *ArgTypes);

/// adds a target shared library to the target execution image
void __tgt_register_lib(struct __tgt_bin_desc *Desc);

/// removes a target shared library from the target execution image
void __tgt_unregister_lib(struct __tgt_bin_desc *Desc);

struct __tgt_bin_desc *register_cubin(const char *);

#ifdef __cplusplus
}
#endif
