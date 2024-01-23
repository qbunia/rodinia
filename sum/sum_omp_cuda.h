#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern REAL sum_kernel(REAL *input, int n, int kernel);
#ifdef __cplusplus
}
#endif
