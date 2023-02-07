#include "rex_kmp.h" 
char OUT__2__5286__main__295__id__ = 0;
struct __tgt_offload_entry OUT__2__5286__main__295__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__2__5286__main__295__id__)), "OUT__2__5286__main__295__kernel__", 0, 0, 0};
char OUT__1__5286__main__244__id__ = 0;
struct __tgt_offload_entry OUT__1__5286__main__244__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__5286__main__244__id__)), "OUT__1__5286__main__244__kernel__", 0, 0, 0};
//====================================================================================================100
//		UPDATE
//====================================================================================================100
//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments
//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"
#define NUM_TEAMS 256
#define NUM_THREADS 1024
//====================================================================================================100
//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100
//====================================================================================================100

int main(int argc,char *argv[])
{
  int status = 0;
//================================================================================80
// 	VARIABLES
//================================================================================80
// time
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;
  long long time8;
  long long time9;
  long long time10;
  time0 = get_time();
// inputs image, input paramenters
  float *image_ori;
// originalinput image
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;
// inputs image, input paramenters
  float *image;
// input image
  long Nr;
  long Nc;
// IMAGE nbr of rows/cols/elements
  long Ne;
// algorithm parameters
  int niter;
// nbr of iterations
  float lambda;
// update step size
// size of IMAGE
  int r1;
  int r2;
  int c1;
  int c2;
// row/col coordinates of uniform ROI
  long NeROI;
// ROI nbr of elements
// ROI statistics
  float meanROI;
  float varROI;
  float q0sqr;
// local region statistics
// surrounding pixel indicies
  int *iN;
  int *iS;
  int *jE;
  int *jW;
// center pixel value
  float Jc;
// directional derivatives
  float *dN;
  float *dS;
  float *dW;
  float *dE;
// calculation variables
  float tmp;
  float sum;
  float sum2;
  float G2;
  float L;
  float num;
  float den;
  float qsqr;
  float D;
// diffusion coefficient
  float *c;
  float cN;
  float cS;
  float cW;
  float cE;
// counters
  int iter;
// primary loop
  long i;
  long j;
// image row/col
  long k;
// image single index
  time1 = get_time();
//================================================================================80
// 	GET INPUT PARAMETERS
//================================================================================80
  if (argc != 5) {
    printf("ERROR: wrong number of arguments\n");
    return 0;
  }
   else {
    niter = atoi(argv[1]);
    lambda = (atof(argv[2]));
    Nr = (atoi(argv[3]));
// it is 502 in the original image
    Nc = (atoi(argv[4]));
// it is 458 in the original image
  }
  time2 = get_time();
//================================================================================80
// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
//================================================================================80
// read image
  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = (image_ori_rows * image_ori_cols);
  image_ori = ((float *)(malloc(sizeof(float ) * image_ori_elem)));
  read_graphics("../../../data/srad/image.pgm",image_ori,image_ori_rows,image_ori_cols,1);
  time3 = get_time();
//================================================================================80
// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
//================================================================================80
  Ne = Nr * Nc;
  image = ((float *)(malloc(sizeof(float ) * Ne)));
  resize(image_ori,image_ori_rows,image_ori_cols,image,Nr,Nc,1);
  time4 = get_time();
//================================================================================80
// 	SETUP
//================================================================================80
  r1 = 0;
// top row index of ROI
  r2 = (Nr - 1);
// bottom row index of ROI
  c1 = 0;
// left column index of ROI
  c2 = (Nc - 1);
// right column index of ROI
// ROI image size
  NeROI = ((r2 - r1 + 1) * (c2 - c1 + 1));
// number of elements in ROI, ROI size
// allocate variables for surrounding pixels
  iN = (malloc(sizeof(int *) * Nr));
// north surrounding element
  iS = (malloc(sizeof(int *) * Nr));
// south surrounding element
  jW = (malloc(sizeof(int *) * Nc));
// west surrounding element
  jE = (malloc(sizeof(int *) * Nc));
// east surrounding element
// allocate variables for directional derivatives
  dN = (malloc(sizeof(float ) * Ne));
// north direction derivative
  dS = (malloc(sizeof(float ) * Ne));
// south direction derivative
  dW = (malloc(sizeof(float ) * Ne));
// west direction derivative
  dE = (malloc(sizeof(float ) * Ne));
// east direction derivative
// allocate variable for diffusion coefficient
  c = (malloc(sizeof(float ) * Ne));
// diffusion coefficient
// N/S/W/E indices of surrounding pixels (every element of IMAGE)
// #pragma omp parallel
  for (i = 0; i < Nr; i++) {
    iN[i] = (i - 1);
// holds index of IMAGE row above
    iS[i] = (i + 1);
// holds index of IMAGE row below
  }
// #pragma omp parallel
  for (j = 0; j < Nc; j++) {
    jW[j] = (j - 1);
// holds index of IMAGE column on the left
    jE[j] = (j + 1);
// holds index of IMAGE column on the right
  }
// N/S/W/E boundary conditions, fix surrounding indices outside boundary of
// IMAGE
  iN[0] = 0;
// changes IMAGE top row index from -1 to 0
  iS[Nr - 1] = (Nr - 1);
// changes IMAGE bottom row index from Nr to Nr-1
  jW[0] = 0;
// changes IMAGE leftmost column index from -1 to 0
  jE[Nc - 1] = (Nc - 1);
// changes IMAGE rightmost column index from Nc to Nc-1
  time5 = get_time();
//================================================================================80
// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
//================================================================================80
// #pragma omp parallel
  for (i = 0; i < Ne; i++) {
// do for the number of elements in input IMAGE
    image[i] = (exp((image[i] / 255)));
// exponentiate input IMAGE and copy to output image
  }
  time6 = get_time();
//================================================================================80
// 	COMPUTATION
//================================================================================80
// printf("iterations: ");
// primary loop
/* Translated from #pragma omp target data ... */
{
    int32_t __arg_num = 10;
    int64_t __arg_types[] = {35, 33, 33, 33, 33, 33, 33, 33, 33, 33};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(float ) * Ne)), ((int64_t )(sizeof(int ) * Nr)), ((int64_t )(sizeof(int ) * Nr)), ((int64_t )(sizeof(int ) * Nc)), ((int64_t )(sizeof(int ) * Nc)), ((int64_t )(sizeof(float ) * Ne)), ((int64_t )(sizeof(float ) * Ne)), ((int64_t )(sizeof(float ) * Ne)), ((int64_t )(sizeof(float ) * Ne)), ((int64_t )(sizeof(float ) * Ne))};
    void *__args[] = {image + 0, iN + 0, iS + 0, jE + 0, jW + 0, dN + 0, dS + 0, dW + 0, dE + 0, c + 0};
    void *__args_base[] = {image, iN, iS, jE, jW, dN, dS, dW, dE, c};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    for (iter = 0; iter < niter; iter++) {
// do for the number of iterations input parameter
// ROI statistics for entire ROI (single number for ROI)
      sum = 0;
      sum2 = 0;
      for (i = r1; i <= r2; i++) {
// do for the range of rows in ROI
        for (j = c1; j <= c2; j++) {
// do for the range of columns in ROI
          tmp = image[i + Nr * j];
// get coresponding value in IMAGE
          sum += tmp;
// take corresponding value and add to sum
          sum2 += tmp * tmp;
// take square of corresponding value and add to sum2
        }
      }
      meanROI = sum / NeROI;
// gets mean (average) value of element in ROI
      varROI = sum2 / NeROI - meanROI * meanROI;
// gets variance of ROI
      q0sqr = varROI / (meanROI * meanROI);
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = 1024;
        int _num_blocks_ = 256;
        void *__host_ptr = (void *)(&OUT__1__5286__main__244__id__);
        void *__args_base[] = {image, &Nr, &Nc, &q0sqr, iN, iS, jE, jW, dN, dS, dW, dE, c};
        void *__args[] = {image, &Nr, &Nc, &q0sqr, iN, iS, jE, jW, dN, dS, dW, dE, c};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(float *))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(float ))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *)))};
        int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33};
        int32_t __arg_num = 13;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
{
/* Launch CUDA kernel ... */
        int64_t __device_id = 0;
        int _threads_per_block_ = 1024;
        int _num_blocks_ = 256;
        void *__host_ptr = (void *)(&OUT__2__5286__main__295__id__);
        void *__args_base[] = {image, &Nr, &Nc, &lambda, iS, jE, dN, dS, dW, dE, c};
        void *__args[] = {image, &Nr, &Nc, &lambda, iS, jE, dN, dS, dW, dE, c};
        int64_t __arg_sizes[] = {((int64_t )(sizeof(float *))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(long ))), ((int64_t )(sizeof(float ))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(int *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *))), ((int64_t )(sizeof(float *)))};
        int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33};
        int32_t __arg_num = 11;
        __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
      }
    }
// primary loop ends
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
// target data region ends
  time7 = get_time();
//================================================================================80
// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
//================================================================================80
// #pragma omp parallel
  for (i = 0; i < Ne; i++) {
// do for the number of elements in IMAGE
    image[i] = (log(image[i]) * 255);
// take logarithm of image, log compress
  }
  time8 = get_time();
//================================================================================80
// 	WRITE IMAGE AFTER PROCESSING
//================================================================================80
  write_graphics("image_out.pgm",image,Nr,Nc,1,255);
  time9 = get_time();
//================================================================================80
// 	DEALLOCATE
//================================================================================80
  free(image_ori);
  free(image);
  free(iN);
  free(iS);
  free(jW);
  free(jE);
// deallocate surrounding pixel memory
  free(dN);
  free(dS);
  free(dW);
  free(dE);
// deallocate directional derivative memory
  free(c);
// deallocate diffusion coefficient memory
  time10 = get_time();
//================================================================================80
//		DISPLAY TIMING
//================================================================================80
  printf("Time spent in different stages of the application:\n");
  printf("%.12f s, %.12f %% : SETUP VARIABLES\n",(((float )(time1 - time0)) / 1000000),(((float )(time1 - time0)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : READ COMMAND LINE PARAMETERS\n",(((float )(time2 - time1)) / 1000000),(((float )(time2 - time1)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : READ IMAGE FROM FILE\n",(((float )(time3 - time2)) / 1000000),(((float )(time3 - time2)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : RESIZE IMAGE\n",(((float )(time4 - time3)) / 1000000),(((float )(time4 - time3)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : SETUP, MEMORY ALLOCATION\n",(((float )(time5 - time4)) / 1000000),(((float )(time5 - time4)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : EXTRACT IMAGE\n",(((float )(time6 - time5)) / 1000000),(((float )(time6 - time5)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : COMPUTE\n",(((float )(time7 - time6)) / 1000000),(((float )(time7 - time6)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : COMPRESS IMAGE\n",(((float )(time8 - time7)) / 1000000),(((float )(time8 - time7)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : SAVE IMAGE INTO FILE\n",(((float )(time9 - time8)) / 1000000),(((float )(time9 - time8)) / ((float )(time10 - time0)) * 100));
  printf("%.12f s, %.12f %% : FREE MEMORY\n",(((float )(time10 - time9)) / 1000000),(((float )(time10 - time9)) / ((float )(time10 - time0)) * 100));
  printf("Total time:\n");
  printf("%.12f s\n",(((float )(time10 - time0)) / 1000000));
//====================================================================================================100
//	END OF FILE
//====================================================================================================100
}
