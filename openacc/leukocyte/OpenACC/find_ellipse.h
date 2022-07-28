#ifndef FIND_ELLIPSE_H
#define FIND_ELLIPSE_H

#include "avilib.h"
#include "matrix.h"
#include "misc_math.h"
#include <math.h>
#include <stdlib.h>

// Defines the region in the video frame containing the blood vessel
#define TOP 110
#define BOTTOM 328
// The number of sample points per ellipse
#define NPOINTS 150
// The number of different sample ellipses to try
#define NCIRCLES 7

extern long long get_time();

extern MAT * get_frame(avi_t *cell_file, int frame_num, int cropped, int scaled);
extern MAT * chop_flip_image(unsigned char *image, int height, int width, int top, int bottom, int left, int right, int scaled);
extern MAT * GICOV(MAT * grad_x, MAT * grad_y);
extern MAT * dilate(MAT * img_in);
extern MAT * linear_interp2(MAT * m, VEC * X, VEC * Y);
extern MAT * TMatrix(unsigned int N, unsigned int M);
extern VEC * getsampling(MAT * m, int ns);
extern VEC * getfdriv(MAT * m, int ns);

extern void compute_constants();
extern void uniformseg(VEC * cellx_row, VEC * celly_row, MAT * x, MAT * y);
extern void splineenergyform01(MAT * Cx, MAT * Cy, MAT * Ix, MAT * Iy, int ns, double delta, double dt, int typeofcell);

extern float * structuring_element(int radius);

extern double m_min(MAT * m);
extern double m_max(MAT * m);

extern float *gicov_mem;
extern float sin_angle[NPOINTS], cos_angle[NPOINTS], theta[NPOINTS];
extern int tX[NCIRCLES * NPOINTS], tY[NCIRCLES * NPOINTS];

extern float *strel;
extern const int radius;
extern const int strel_m;
extern const int strel_n;

#endif
