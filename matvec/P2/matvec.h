#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void matvec_cuda(REAL *result, REAL *vector, REAL * matrix, int n, int m);
#ifdef __cplusplus
}
#endif
