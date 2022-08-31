#include <stdio.h>

void lud_openacc(float *a, int size)
{
     int i,j,k;
     float sum;
     #pragma acc data copy(a[0:size*size])
     for (i=0; i <size; i++){
         #pragma acc parallel loop
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             #pragma acc loop seq
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }
         #pragma acc parallel loop
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             #pragma acc loop seq
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
