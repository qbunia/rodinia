#include <stdio.h>

void lud_openmp_gpu(float *a, int size)
{
     int i,j,k;
     float sum;

     #pragma omp target teams distribute map(tofrom:a[0:size*size]) private(i,j,k) 
     for (i=0; i <size; i++){
          
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             //improve 10% performance
             #pragma omp parallel for reduction(+:sum)
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
