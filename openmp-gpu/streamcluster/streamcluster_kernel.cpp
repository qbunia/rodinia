/***********************************************
	streamcluster_kernel.cpp
	: parallelized code of streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/
#include "streamcluster.h"

using namespace std;

static int iter = 0;		// counter for total# of iteration


//=======================================
// Euclidean Distance
//=======================================
#define D_DIST(p1,p2,num,dim,coord,ret) \
{ \
	ret = 0.0; \
	for(int i = 0; i < dim; i++){ \
		float tmp = coord[(i*num)+p1] - coord[(i*num)+p2]; \
		ret += tmp * tmp; \
	} \
}


//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, 
			int *center_table, bool *switch_membership, bool isCoordChanged)
{	
	float tmp_t;

	const int TRANSFER_CENTER_TABLE		= 1;
	const int TRNASFER_COORD			= 2;
	const int TRANSFER_POINTS			= 3;
	const int TRANSFER_SWITCH_MEMSHIP 	= 4;

	int stride	= *numcenters + 1;			// size of each work_mem segment
	int K		= *numcenters ;				// number of centers
	int num		=  points->num;				// number of points
	int dim		=  points->dim;				// number of dimension
	int nThread =  num;						// number of threads == number of data points
	
	//=========================================
	// INITIALIZATION AND DATA PREPARATION
	//=========================================
	// build center-index table
	int count = 0;
	for( int i=0; i<num; i++)
	{
		if( is_center[i] )
		{
			center_table[i] = count++;
		}
	}
	#pragma acc update device(center_table) async(TRANSFER_CENTER_TABLE)

	// Extract 'coord'
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		for(int i=0; i<dim; i++)
		{
			for(int j=0; j<num; j++)
			{
				coord[ (i*num)+j ] = points->p[j].coord[i];
			}
		}
		#pragma acc update device(coord[0:num*dim]) async(TRNASFER_COORD)
	}

	#pragma acc update device(points->p) async(TRANSFER_POINTS)

	#pragma acc kernels
	for(int i=0; i<num; i++) {
		switch_membership[i] = 0;
	}

	#pragma acc kernels
	for(int i=0; i<stride * (nThread + 1); i++) {
		work_mem[i] = 0;
	}
	
	*serial_t += (double) tmp_t;	
	*alloc_t += (double) tmp_t;

	#pragma acc wait(TRANSFER_CENTER_TABLE)
	if(isCoordChanged || iter == 0) {
		#pragma acc wait(TRNASFER_COORD)
	}
	#pragma acc wait(TRANSFER_POINTS)
	
	//=======================================
	// KERNEL: CALCULATE COST
	#pragma acc kernels
	for(int i=0; i<num; i++)
	{
		float *lower = &work_mem[i*stride];
		
		// cost between this point and point[x]: euclidean distance multiplied by weight
		float x_cost = D_DIST(i, x, num, dim, coord) * points->p[i].weight;
		
		// if computed cost is less then original (it saves), mark it as to reassign
		if ( x_cost < p[i].cost )
		{
			switch_membership[i] = 1;
			lower[K] += x_cost - points->p[i].cost;
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table[points->p[i].assign]] += points->p[i].cost - x_cost;
		}
	}

	*kernel_t += (double) tmp_t;
	
	//=======================================
	// GPU-TO-CPU MEMORY COPY
	//=======================================
	#pragma acc update host(work_mem[0:])
	#pragma acc update host(switch_membership[0:num]) async(TRANSFER_SWITCH_MEMSHIP)

	//=======================================
	// CPU (SERIAL) WORK
	//=======================================
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem[stride * nThread];
	// compute the number of centers to close if we are to open i
	for(int i=0; i < num; i++)
	{
		if( is_center[i] )
		{
			float low = z;
		    for( int j = 0; j < num; j++ )
			{
				low += work_mem[ j*stride + center_table[i] ];
			}
			
		    gl_lower[center_table[i]] = low;
				
		    if ( low > 0 )
			{
				++number_of_centers_to_close;
				work_mem[i*stride+K] -= low;
		    }
		}
		gl_cost_of_opening_x += work_mem[i*stride+K];
	}

	#pragma acc wait(TRANSFER_SWITCH_MEMSHIP)

	//if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
	if ( gl_cost_of_opening_x < 0 )
	{
		for(int i = 0; i < num; i++)
		{
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
			if ( switch_membership[i] || close_center )
			{
				points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
				points->p[i].assign = x;
			}
		}
		
		for(int i = 0; i < num; i++)
		{
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
			{
				is_center[i] = false;
			}
		}
		
		if( x >= 0 && x < num)
		{
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	}
	else
	{
		gl_cost_of_opening_x = 0;
	}

	iter++;
	return -gl_cost_of_opening_x;
}
