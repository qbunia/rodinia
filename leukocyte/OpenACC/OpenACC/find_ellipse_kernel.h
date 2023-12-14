#ifndef _FIND_ELLIPSE_KERNEL_H_
#define _FIND_ELLIPSE_KERNEL_H_

extern float *gicov_kernel(int grad_m, int grad_n, 
	                       float *host_grad_x, float *host_grad_y, 
			               float *gicov);
extern float *dilate_kernel(float *gicov, float *img_dilated, 
	                        int max_gicov_m, int max_gicov_n, 
	                        int strel_m, int strel_n);

#endif
