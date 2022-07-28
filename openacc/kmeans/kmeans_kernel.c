#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "kmeans.h"

int setup(int argc, char** argv);									/* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *block_clusters_d;											/* per block calculation of cluster centers */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{
	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*) malloc(npoints * sizeof(int));
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));

	/* invert the data array (kernel execution) */	
	invert_mapping(feature_flipped_d,feature_d,npoints,nfeatures);
}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
}
/* -------------- deallocateMemory() end ------------------- */



////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
	setup(argc, argv);    
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ... 
   [p1,dim0][p1,dim1][p1,dim2] ... 
   [p2,dim0][p2,dim1][p2,dim2] ... 
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
void invert_mapping(float *input,			/* original */
				   float *output,			/* inverted */
				   int npoints,				/* npoints */
				   int nfeatures)			/* nfeatures */
{
	int point_id,i;

	#pragma acc kernels
	for(point_id=0;point_id<npoints;point_id++)
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
}
/* ----------------- invert_mapping() end --------------------- */

/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters) 
{
	int point_id;
	
	#pragma acc kernels
	for(point_id=0;point_id<npoints;point_id++) {
		int i, j;
		int index = -1;												/* index of closest cluster center id */
		float min_dist = FLT_MAX;
		float dist;													/* distance square between a point to cluster center */
		
		/* find the cluster center id with min distance to pt */
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */			
			float ans=0.0;											/* Euclidean distance sqaure */

			for (j=0; j < nfeatures; j++)
			{					
				int addr = point_id + j*npoints;					/* appropriate index of data point */
				float diff = (tex1Dfetch(t_features,addr) -
							  clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
				ans += diff*diff;									/* sum of squares */
			}
			dist = ans;		

			/* see if distance is smaller than previous ones:
			if so, change minimum distance and save index of cluster center */
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}

		/* assign the membership to object point_id */
		membership[point_id] = index;
	}
}


/* ------------------- kmeans() ------------------------ */    
int	// delta -- had problems when return value was of float type
kmeans(float   *feature,				/* in: [npoints][nfeatures] */
       int      nfeatures,				/* number of attributes for each point */
       int      npoints,				/* number of data points */
       int      nclusters,				/* number of clusters */
       int     *membership,				/* which cluster the point belongs to */
	   int     *membership_new,         /* newly assignment membership */
	   float  **clusters,				/* coordinates of cluster centers */
	   int     *new_centers_len,		/* number of elements in each cluster */
       float  **new_centers				/* sum of elements in each cluster */
	   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

	/* copy clusters (host to device) */
	#pragma acc update device(clusters[0][0:nclusters*nfeatures])

	/* execute the kernel */
    kmeansPoint( feature_d,
                 nfeatures,
                 npoints,
                 nclusters,
                 membership_new,
                 clusters,
				 block_clusters_d);

	/* copy back membership (device to host) */
	#pragma acc update host(membership_new[0:npoints])

	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
			delta++;
			membership[i] = membership_new[i];
		}
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
	}

	return delta;
}
/* ------------------- kmeans() end ------------------------ */    

