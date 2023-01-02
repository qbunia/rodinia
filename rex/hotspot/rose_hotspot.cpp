#include "rex_kmp.h" 
char OUT__2__5640__single_iteration__113__id__ = 0;
struct __tgt_offload_entry OUT__2__5640__single_iteration__113__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__2__5640__single_iteration__113__id__)), "OUT__2__5640__single_iteration__113__kernel__", 0, 0, 0};
char OUT__1__5640__single_iteration__33__id__ = 0;
struct __tgt_offload_entry OUT__1__5640__single_iteration__33__omp_offload_entry__ __attribute__((section("omp_offloading_entries")))  = {((void *)(&OUT__1__5640__single_iteration__33__id__)), "OUT__1__5640__single_iteration__33__kernel__", 0, 0, 0};
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;
#define STR_SIZE 256
/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
/* chip parameters	*/
double t_chip = 0.0005;
double chip_height = 0.016;
double chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
double amb_temp = 80.0;
/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */

void single_iteration(double *result,double *temp,double *power,int row,int col,double Cap,double Rx,double Ry,double Rz,double step)
{
  double delta;
  int r;
  int c;
{
// Launch CUDA kernel ...
    int _threads_per_block_ = 128;
    int _num_blocks_ = 256;
    int64_t __device_id = 0;
    void *__host_ptr = (void *)(&OUT__1__5640__single_iteration__33__id__);
    void *__args_base[] = {&row, &col, &Cap, &Rx, &Ry, &Rz, &step, &delta, result, temp, power};
    void *__args[] = {&row, &col, &Cap, &Rx, &Ry, &Rz, &step, &delta, result + 0, temp + 0, power + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ))), ((int64_t )(sizeof(double ) * (row * col))), ((int64_t )(sizeof(double ) * (row * col))), ((int64_t )(sizeof(double ) * (row * col)))};
    int64_t __arg_types[] = {33, 33, 33, 33, 33, 33, 33, 33, 34, 35, 33};
    int32_t __arg_num = 11;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
{
// Launch CUDA kernel ...
    int _threads_per_block_ = 128;
    int _num_blocks_ = 256;
    int64_t __device_id = 0;
    void *__host_ptr = (void *)(&OUT__2__5640__single_iteration__113__id__);
    void *__args_base[] = {&row, &col, result, temp};
    void *__args[] = {&row, &col, result + 0, temp + 0};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(int ))), ((int64_t )(sizeof(int ))), ((int64_t )(sizeof(double ) * (row * col))), ((int64_t )(sizeof(double ) * (row * col)))};
    int64_t __arg_types[] = {33, 33, 34, 35};
    int32_t __arg_num = 4;
    __tgt_target_teams(__device_id,__host_ptr,__arg_num,__args_base,__args,__arg_sizes,__arg_types,_num_blocks_,_threads_per_block_);
  }
}
/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */

void compute_tran_temp(double *result,int num_iterations,double *temp,double *power,int row,int col)
{
#ifdef VERBOSE
#endif
  double grid_height = chip_height / row;
  double grid_width = chip_width / col;
  double Cap = 0.5 * 1.75e6 * t_chip * grid_width * grid_height;
  double Rx = grid_width / (2.0 * 100 * t_chip * grid_height);
  double Ry = grid_height / (2.0 * 100 * t_chip * grid_width);
  double Rz = t_chip / (100 * grid_height * grid_width);
  double max_slope = 3.0e6 / (0.5 * t_chip * 1.75e6);
  double step = 0.001 / max_slope;
#ifdef VERBOSE
#endif
// Translated from #pragma omp target data ...
{
    int32_t __arg_num = 3;
    int64_t __arg_types[] = {34, 35, 33};
    int64_t __arg_sizes[] = {((int64_t )(sizeof(double ) * (row * col))), ((int64_t )(sizeof(double ) * (row * col))), ((int64_t )(sizeof(double ) * (row * col)))};
    void *__args[] = {result + 0, temp + 0, power + 0};
    void *__args_base[] = {result, temp, power};
    int64_t __device_id = 0;
    __tgt_target_data_begin(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
    for (int i = 0; i < num_iterations; i++) {
#ifdef VERBOSE
#endif
      single_iteration(result,temp,power,row,col,Cap,Rx,Ry,Rz,step);
    }
    __tgt_target_data_end(__device_id,__arg_num,__args_base,__args,__arg_sizes,__arg_types);
  }
/* end pragma target data */
#ifdef VERBOSE
#endif
}

void fatal(const char *s)
{
  fprintf(stderr,"error: %s\n",s);
  exit(1);
}

void read_input(double *vect,int grid_rows,int grid_cols,char *file)
{
  int i;
  FILE *fp;
  char str[256];
  double val;
  fp = fopen(file,"r");
  if (!fp) 
    fatal("file could not be opened for reading");
  for (i = 0; i < grid_rows * grid_cols; i++) {
    fgets(str,256,fp);
    if ((feof(fp))) 
      fatal("not enough lines in file");
    if (sscanf(str,"%lf",&val) != 1) 
      fatal("invalid file format");
    vect[i] = val;
  }
  fclose(fp);
}

void usage(int argc,char **argv)
{
  fprintf(stderr,"Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file>\n",argv[0]);
  fprintf(stderr,"\t<grid_rows>  - number of rows in the grid (positive integer)\n");
  fprintf(stderr,"\t<grid_cols>  - number of columns in the grid (positive integer)\n");
  fprintf(stderr,"\t<sim_time>   - number of iterations\n");
  fprintf(stderr,"\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr,"\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  exit(1);
}

int main(int argc,char **argv)
{
  int status = 0;
  int grid_rows;
  int grid_cols;
  int sim_time;
  double *temp;
  double *power;
  double *result;
  char *tfile;
  char *pfile;
/* check validity of inputs	*/
  if (argc != 6) 
    usage(argc,argv);
  if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 || (sim_time = atoi(argv[3])) <= 0) 
    usage(argc,argv);
/* allocate memory for the temperature and power arrays	*/
  temp = ((double *)(calloc((grid_rows * grid_cols),sizeof(double ))));
  power = ((double *)(calloc((grid_rows * grid_cols),sizeof(double ))));
  result = ((double *)(calloc((grid_rows * grid_cols),sizeof(double ))));
  if (!temp || !power) 
    fatal("unable to allocate memory");
/* read initial temperatures and input power	*/
  tfile = argv[4];
  pfile = argv[5];
  read_input(temp,grid_rows,grid_cols,tfile);
  read_input(power,grid_rows,grid_cols,pfile);
  printf("Start computing the transient temperature\n");
  compute_tran_temp(result,sim_time,temp,power,grid_rows,grid_cols);
  printf("Ending simulation\n");
/* output results	*/
#ifdef VERBOSE
#endif
#ifdef OUTPUT
#endif
/* cleanup	*/
  free(temp);
  free(power);
  return 0;
}
