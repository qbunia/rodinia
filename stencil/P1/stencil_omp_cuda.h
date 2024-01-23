#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void stencil_kernel(REAL*, REAL*, int, int, const float*, int, int, int);
#ifdef __cplusplus
}
#endif
