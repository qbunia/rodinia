#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <omp.h>									// (in path known to compiler)			needed by openmp
#include <stdlib.h>									// (in path known to compiler)			needed by malloc
#include <stdio.h>									// (in path known to compiler)			needed by printf
#include <math.h>									// (in path known to compiler)			needed by exp

#define NUM_THREADS 128

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)	needed to recognized input variables

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer

//======================================================================================================================================================150
//	KERNEL_CPU FUNCTION HEADER
//======================================================================================================================================================150

#include "kernel_gpu.h"								// (in the current directory)

//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200

void  kernel_gpu(	par_str par, 
					dim_str dim,
					box_str* box,
					FOUR_VECTOR* rv,
					fp* qv,
					FOUR_VECTOR* fv
                       
          )
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;

	time0 = get_time();

	// timer
	long long time1;
	long long time2;
	long long time3;
	long long time4;

	// parameters
	fp alpha;
	fp a2;

	// counters
	int i, j, k, l;

	// home box
	long first_i;
	FOUR_VECTOR* rA;
	FOUR_VECTOR* fA;

	// neighbor box
	int pointer;
	long first_j; 
	FOUR_VECTOR* rB;
	fp* qB;

	// common
	fp r2; 
	fp u2;
	fp fs;
	fp vij;
	fp fxij,fyij,fzij;
	THREE_VECTOR d;

	time1 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	omp_set_num_threads(dim.cores_arg);

	time2 = get_time();

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	alpha = par.alpha;
	a2 = 2.0*alpha*alpha;

	time3 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

//	#pragma omp	parallel for \
	//			private(i, j, k) \
	//			private(first_i, rA, fA) \
//				private(pointer, first_j, rB, qB) \
	//			private(r2, u2, fs, vij, fxij, fyij, fzij, d)
 /*
 void  kernel_cpu(	par_str par, 
					dim_str dim,
					box_str* box,
					FOUR_VECTOR* rv,
					fp* qv,
					FOUR_VECTOR* fv
                       
          )
          */
 #pragma acc data copy(fv[0:dim.space_elem])
 #pragma acc data copyin(par, dim) copyin(rv[0:dim.space_elem], qv[0:dim.space_elem]) copyin(box[0:dim.number_boxes])
/* #pragma acc parallel loop \
        num_gangs(dim.number_boxes) num_workers(NUM_THREADS) \
        private(i, j, k) \
				private(first_i, rA, fA) \
				private(pointer, first_j, rB, qB) \
				private(r2, u2, fs, vij, fxij, fyij, fzij, d)
*/
  #pragma acc parallel loop \
        num_gangs(dim.number_boxes) num_workers(1) vector_length(NUM_THREADS) \
        private(i, j, k) \
				private(first_i, rA, fA) \
				private(pointer, first_j, rB, qB) \
				private(r2, u2, fs, vij, fxij, fyij, fzij, d)
	for(l=0; l<dim.number_boxes; l=l+1){
		//------------------------------------------------------------------------------------------100
		//	home box - box parameters
		//------------------------------------------------------------------------------------------100

		first_i = box[l].offset;												// offset to common arrays

		//------------------------------------------------------------------------------------------100
		//	home box - distance, force, charge and type parameters from common arrays
		//------------------------------------------------------------------------------------------100

		rA = &rv[first_i];
		fA = &fv[first_i];

		//------------------------------------------------------------------------------------------100
		//	Do for the # of (home+neighbor) boxes
		//------------------------------------------------------------------------------------------100

		for (k=0; k<(1+box[l].nn); k++) 
		{

			//----------------------------------------50
			//	neighbor box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = l;													// set first box to be processed to home box
			}
			else{
				pointer = box[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//----------------------------------------50
			//	neighbor box - box parameters
			//----------------------------------------50

			first_j = box[pointer].offset; 

			//----------------------------------------50
			//	neighbor box - distance, force, charge and type parameters
			//----------------------------------------50

			rB = &rv[first_j];
			qB = &qv[first_j];

			//----------------------------------------50
			//	Do for the # of particles in home box
			//----------------------------------------50

			for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1){

				// do for the # of particles in current (home or neighbor) box
				for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1){

					// // coefficients
					r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2.*vij;
					d.x = rA[i].x  - rB[j].x; 
					d.y = rA[i].y  - rB[j].y; 
					d.z = rA[i].z  - rB[j].z; 
					fxij=fs*d.x;
					fyij=fs*d.y;
					fzij=fs*d.z;

					// forces
					fA[i].v +=  qB[j]*vij;
					fA[i].x +=  qB[j]*fxij;
					fA[i].y +=  qB[j]*fyij;
					fA[i].z +=  qB[j]*fzij;

				} // for j

			} // for i

		} // for k

	} // for l

	time4 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : CPU/MCPU: VARIABLES\n",				(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: INPUTS\n", 				(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time4-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time4-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time4-time0) / 1000000);

} // main

#ifdef __cplusplus
}
#endif
