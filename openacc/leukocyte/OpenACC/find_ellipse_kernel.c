#include "find_ellipse_kernel.h"
#include <stdio.h>


// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)


// Sets up and invokes the GICOV kernel and returns its output
float *gicov_kernel(int grad_m, int grad_n, 
	                float *host_grad_x, float *host_grad_y, 
			        float *gicov) {

	int i, j;
    
	// Execute the GICOV kernel
	#pragma acc kernels loop
	for (i = 0; i < grad_m; i++) {
		for (j = 0; j < grad_n; j++) {

			int k, n, x, y;

			// Initialize the maximal GICOV score to 0
			float max_GICOV = 0.f;

			// Iterate across each stencil
			for (k = 0; k < NCIRCLES; k++) {
				// Variables used to compute the mean and variance
				//  of the gradients along the current stencil
				float sum = 0.f, M2 = 0.f, mean = 0.f;		
				
				// Iterate across each sample point in the current stencil
				for (n = 0; n < NPOINTS; n++) {
					// Determine the x- and y-coordinates of the current sample point
					y = j + tY[(k * NPOINTS) + n];
					x = i + tX[(k * NPOINTS) + n];
					
					// Compute the combined gradient value at the current sample point
					float p = grad_x[x * grad_m + y] * cos_angle[n] +
							  grad_y[x * grad_m + y] * sin_angle[n];
					
					// Update the running total
					sum += p;
					
					// Partially compute the variance
					float delta = p - mean;
					mean = mean + (delta / (float) (n + 1));
					M2 = M2 + (delta * (p - mean));
				}
				
				// Finish computing the mean
				mean = sum / ((float) NPOINTS);
				
				// Finish computing the variance
				float var = M2 / ((float) (NPOINTS - 1));
				
				// Keep track of the maximal GICOV value seen so far
				if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
			}
			
			// Store the maximal GICOV value
			gicov[(i * grad_m) + j] = max_GICOV;
		}
	}
}


// Sets up and invokes the dilation kernel and returns its output
float *dilate_kernel(float *gicov, float *img_dilated, 
	                 int max_gicov_m, int max_gicov_n, 
	                 int strel_m, int strel_n) {
	int i,j;

	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Execute the dilation kernel
	#pragma acc kernels loop
	for (i = 0; i < max_gicov_n; i++) {
		for (j = 0; j < max_gicov_m; j++) {
			// Initialize the maximum GICOV score seen so far to zero
			float max = 0.0;

			// Iterate across the structuring element in one dimension
			int el_i, el_j, x, y;
			for(el_i = 0; el_i < strel_m; el_i++) {
				y = i - el_center_i + el_i;
				// Make sure we have not gone off the edge of the matrix
				if( (y >= 0) && (y < max_gicov_m) ) {
					// Iterate across the structuring element in the other dimension
					for(el_j = 0; el_j < strel_n; el_j++) {
						x = j - el_center_j + el_j;
						// Make sure we have not gone off the edge of the matrix
						//  and that the current structuring element value is not zero
						if( (x >= 0) &&
							(x < max_gicov_n) &&
							(strel[(el_i * strel_n) + el_j] != 0) ) {
								// Determine if this is maximal value seen so far
								float temp = gicov[(x * max_gicov_m) + y];
								if (temp > max) max = temp;
						}
					}
				}
			}
			
			// Store the maximum value found
			img_dilated[(i * max_gicov_n) + j] = max;
		}
	}
}
