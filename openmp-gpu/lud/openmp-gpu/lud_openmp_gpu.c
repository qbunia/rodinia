#include <stdio.h>
#define NUM_TEAMS 256
#define NUM_THREADS 1024

void lud_openmp_gpu(float *a, int size)
{
     int i,j,k;
     float sum;
     #pragma omp target data map(tofrom:a[0:size*size])
     
     for (i=0; i <size; i++){
         #pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }
         #pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
            // #pragma omp for
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
