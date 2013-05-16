//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

//======================================================================================================================================================
//	LIBRARIES
//======================================================================================================================================================

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <avilib.h>
#include <avimod.h>

//======================================================================================================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//======================================================================================================================================================

#include "define.c"

params_common_change common_change;
params_common common;


//  POINT
int* Row[ALL_POINTS];
int* Col[ALL_POINTS];
int* tRowLoc[ALL_POINTS];
int* tColLoc[ALL_POINTS];
fp* d_T[ALL_POINTS];
//  POINT NUMBER
int point_no[ALL_POINTS];
//  RIGHT TEMPLATE  FROM    TEMPLATE ARRAY
int in_pointer[ALL_POINTS];
//  AREA AROUND POINT       FROM    FRAME
fp** d_in2;
//  CONVOLUTION
fp** d_conv;
fp* d_in_mod; ???
//  PAD ARRAY, VERTICAL CUMULATIVE SUM
fp** d_in2_pad_cumv;
//  SELECTION
fp** d_in2_pad_cumv_sel;
//  SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
fp** d_in2_sub_cumh;
//  SELECTION
fp** d_in2_sub_cumh_sel;
//  SELECTION 2, SUBTRACTION
fp** d_in2_sub2;
//  MULTIPLICATION
fp** d_in2_sqr;
//  SELECTION 2, SUBTRACTION
fp** d_in2_sqr_sub2;
//  FINAL
fp** d_in_sqr;
//  TEMPLATE MASK
fp** d_tMask;
//  MASK CONVOLUTION
fp** d_mask_conv;


//======================================================================================================================================================
// KERNEL CODE
//======================================================================================================================================================

#include "kernel.c"

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

int main(int argc, char *argv []){

    //======================================================================================================================================================
    //	VARIABLES
    //======================================================================================================================================================

    // counter
    int i;
    int frames_processed;

    // frames
    char* video_file_name;
    avi_t* frames;
    fp* frame;

    //======================================================================================================================================================
    // 	FRAME
    //======================================================================================================================================================

    if(argc!=3){
        printf("ERROR: usage: heartwall <inputfile> <num of frames>\n");
        exit(1);
    }

    // open movie file
    video_file_name = argv[1];
    frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
    if (frames == NULL)  {
           AVI_print_error((char *) "Error with AVI_open_input_file");
           return -1;
    }

#pragma acc data \
    create(common_change.frame[0:common.frame_elem]) \
    create(common.endoRow[0:common.endo_mem]) \
    create(common.endoCol[0:common.endo_mem]) \
    copyout(common.tEndoRowLoc[0:common.endo_mem]) \
    copyout(common.tEndoColLoc[0:common.endo_mem]) \
    create(common.epiRow[0:common.epiPoints]) \
    create(common.epiCol[0:common.epiPoints]) \
    copyout(common.tEpiRowLoc[0:common.epiPoints*common.no_frames]) \
    copyout(common.tEpiColLoc[0:common.epiPoints*common.no_frames]) \
    create(common.endoT[0:common.in_elem*common.endoPoints]) \
    create(common.epiT[0:common.in_elem*common.endoPoints])
{
    // common
    common.no_frames = AVI_video_frames(frames);
    common.frame_rows = AVI_video_height(frames);
    common.frame_cols = AVI_video_width(frames);
    common.frame_elem = common.frame_rows * common.frame_cols;
    common.frame_mem = sizeof(fp) * common.frame_elem;

    // pointers
    common_change.frame = (fp*) malloc(common.frame_mem);

    //======================================================================================================================================================
    // 	CHECK INPUT ARGUMENTS
    //======================================================================================================================================================

    frames_processed = atoi(argv[2]);
        if(frames_processed<0 || frames_processed>common.no_frames){
            printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common.no_frames);
            return 0;
    }


    //======================================================================================================================================================
    //	HARDCODED INPUTS FROM MATLAB
    //======================================================================================================================================================

    //====================================================================================================
    //	CONSTANTS
    //====================================================================================================

    common.sSize = 40;
    common.tSize = 25;
    common.maxMove = 10;
    common.alpha = 0.87;

    //====================================================================================================
    //	ENDO POINTS
    //====================================================================================================

    common.endoPoints = ENDO_POINTS;
    common.endo_mem = sizeof(int) * common.endoPoints;

    common.endoRow = (int *)malloc(common.endo_mem);
    common.endoRow[ 0] = 369;
    common.endoRow[ 1] = 400;
    common.endoRow[ 2] = 429;
    common.endoRow[ 3] = 452;
    common.endoRow[ 4] = 476;
    common.endoRow[ 5] = 486;
    common.endoRow[ 6] = 479;
    common.endoRow[ 7] = 458;
    common.endoRow[ 8] = 433;
    common.endoRow[ 9] = 404;
    common.endoRow[10] = 374;
    common.endoRow[11] = 346;
    common.endoRow[12] = 318;
    common.endoRow[13] = 294;
    common.endoRow[14] = 277;
    common.endoRow[15] = 269;
    common.endoRow[16] = 275;
    common.endoRow[17] = 287;
    common.endoRow[18] = 311;
    common.endoRow[19] = 339;
    #pragma update target(common.endoRow[0:common.endo_mem])

    common.endoCol = (int *)malloc(common.endo_mem);
    common.endoCol[ 0] = 408;
    common.endoCol[ 1] = 406;
    common.endoCol[ 2] = 397;
    common.endoCol[ 3] = 383;
    common.endoCol[ 4] = 354;
    common.endoCol[ 5] = 322;
    common.endoCol[ 6] = 294;
    common.endoCol[ 7] = 270;
    common.endoCol[ 8] = 250;
    common.endoCol[ 9] = 237;
    common.endoCol[10] = 235;
    common.endoCol[11] = 241;
    common.endoCol[12] = 254;
    common.endoCol[13] = 273;
    common.endoCol[14] = 300;
    common.endoCol[15] = 328;
    common.endoCol[16] = 356;
    common.endoCol[17] = 383;
    common.endoCol[18] = 401;
    common.endoCol[19] = 411;
    #pragma acc update target(common.endoCol[0:common.endo_mem])

    //====================================================================================================
    //	EPI POINTS
    //====================================================================================================

    common.epiPoints = EPI_POINTS;
    common.epi_mem = sizeof(int) * common.epiPoints;

    common.epiRow = (int *)malloc(common.epi_mem);
    common.epiRow[ 0] = 390;
    common.epiRow[ 1] = 419;
    common.epiRow[ 2] = 448;
    common.epiRow[ 3] = 474;
    common.epiRow[ 4] = 501;
    common.epiRow[ 5] = 519;
    common.epiRow[ 6] = 535;
    common.epiRow[ 7] = 542;
    common.epiRow[ 8] = 543;
    common.epiRow[ 9] = 538;
    common.epiRow[10] = 528;
    common.epiRow[11] = 511;
    common.epiRow[12] = 491;
    common.epiRow[13] = 466;
    common.epiRow[14] = 438;
    common.epiRow[15] = 406;
    common.epiRow[16] = 376;
    common.epiRow[17] = 347;
    common.epiRow[18] = 318;
    common.epiRow[19] = 291;
    common.epiRow[20] = 275;
    common.epiRow[21] = 259;
    common.epiRow[22] = 256;
    common.epiRow[23] = 252;
    common.epiRow[24] = 252;
    common.epiRow[25] = 257;
    common.epiRow[26] = 266;
    common.epiRow[27] = 283;
    common.epiRow[28] = 305;
    common.epiRow[29] = 331;
    common.epiRow[30] = 360;
    #pragma acc update target(common.epiRow[0:common.epiPoints])

    common.epiCol = (int *)malloc(common.epi_mem);
    common.epiCol[ 0] = 457;
    common.epiCol[ 1] = 454;
    common.epiCol[ 2] = 446;
    common.epiCol[ 3] = 431;
    common.epiCol[ 4] = 411;
    common.epiCol[ 5] = 388;
    common.epiCol[ 6] = 361;
    common.epiCol[ 7] = 331;
    common.epiCol[ 8] = 301;
    common.epiCol[ 9] = 273;
    common.epiCol[10] = 243;
    common.epiCol[11] = 218;
    common.epiCol[12] = 196;
    common.epiCol[13] = 178;
    common.epiCol[14] = 166;
    common.epiCol[15] = 157;
    common.epiCol[16] = 155;
    common.epiCol[17] = 165;
    common.epiCol[18] = 177;
    common.epiCol[19] = 197;
    common.epiCol[20] = 218;
    common.epiCol[21] = 248;
    common.epiCol[22] = 276;
    common.epiCol[23] = 304;
    common.epiCol[24] = 333;
    common.epiCol[25] = 361;
    common.epiCol[26] = 391;
    common.epiCol[27] = 415;
    common.epiCol[28] = 434;
    common.epiCol[29] = 448;
    common.epiCol[30] = 455;
    #pragma acc update target(common.epiCol[0:common.epiPoints)

    //====================================================================================================
    //	ALL POINTS
    //====================================================================================================

    common.allPoints = ALL_POINTS;

    //======================================================================================================================================================
    // 	TEMPLATE SIZES
    //======================================================================================================================================================

    // common
    common.in_rows = common.tSize + 1 + common.tSize;
    common.in_cols = common.in_rows;
    common.in_elem = common.in_rows * common.in_cols;
    common.in_mem = sizeof(fp) * common.in_elem;

    //======================================================================================================================================================
    // 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
    //======================================================================================================================================================

    // common
    common.endoT = (fp*) malloc(common.in_mem * common.endoPoints);
    common.epiT = (fp*) malloc(common.in_mem * common.endoPoints);

    //======================================================================================================================================================
    //	SPECIFIC TO ENDO OR EPI TO BE SET HERE
    //======================================================================================================================================================

    #pragma acc kernels
    for(i=0; i<common.endoPoints; i++){
        unique[i].point_no = i;
        unique[i].Row = common.endoRow;
        unique[i].Col = common.endoCol;
        unique[i].tRowLoc = common.tEndoRowLoc;
        unique[i].tColLoc = common.tEndoColLoc;
        unique[i].d_T = common.endoT;
    }
    #pragma acc kernels
    for(i=common.endoPoints; i<common.allPoints; i++){
        unique[i].point_no = i-common.endoPoints;
        unique[i].Row = common.epiRow;
        unique[i].Col = common.epiCol;
        unique[i].tRowLoc = common.tEpiRowLoc;
        unique[i].tColLoc = common.tEpiColLoc;
        unique[i].d_T = common.epiT;
    }

    //======================================================================================================================================================
    // 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
    //======================================================================================================================================================

    // pointers
    #pragma acc kernels
    for(i=0; i<common.allPoints; i++){
        unique[i].in_pointer = unique[i].point_no * common.in_elem;
    }

    //======================================================================================================================================================
    // 	AREA AROUND POINT		FROM	FRAME
    //======================================================================================================================================================

    // common
    common.in2_rows = 2 * common.sSize + 1;
    common.in2_cols = 2 * common.sSize + 1;
    common.in2_elem = common.in2_rows * common.in2_cols;
    common.in2_mem = sizeof(fp) * common.in2_elem;

    // pointers
    d_in2 = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2[0] = (fp*) malloc(common.allPoints * common.in2_mem);
    for(i=1; i<common.allPoints; i++) {
        d_in2[i] = d_in2[i-1] + common.in2_elem;
    }

    //======================================================================================================================================================
    // 	CONVOLUTION
    //======================================================================================================================================================

    // common
    common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
    common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
    common.conv_elem = common.conv_rows * common.conv_cols;												// number of elements
    common.conv_mem = sizeof(fp) * common.conv_elem;
    common.ioffset = 0;
    common.joffset = 0;

    // pointers
    d_conv = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_conv[0] = (fp*) malloc(common.allPoints * common.conv_mem);
    for(i=1; i<common.allPoints; i++){
        d_conv[i] = d_conv[i-1] + common.conv_elem;
    }

    //======================================================================================================================================================
    // 	CUMULATIVE SUM
    //======================================================================================================================================================

    //====================================================================================================
    // 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
    //====================================================================================================

    // common
    common.in2_pad_add_rows = common.in_rows;
    common.in2_pad_add_cols = common.in_cols;

    common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
    common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
    common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
    common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

    // pointers
    d_in2_pad_cumv = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_pad_cumv[0] = (fp*) malloc(common.allPoints * common.in2_pad_cumv_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_pad_cumv[i] = d_in2_pad_cumv[i-1] + common.in2_pad_cumv_elem;
    }

    //====================================================================================================
    // 	SELECTION
    //====================================================================================================

    // common
    common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
    common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
    common.in2_pad_cumv_sel_collow = 1;
    common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
    common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
    common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
    common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
    common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

    // pointers
    d_in2_pad_cumv_sel = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_pad_cumv_sel[0] = (fp*) malloc(common.allPoints * common.in2_pad_cumv_sel_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_pad_cumv_sel[i] = d_in2_pad_cumv_sel[i-1] + common.in2_pad_cumv_sel_elem;
    }

    //====================================================================================================
    // 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
    //====================================================================================================

    // common
    common.in2_pad_cumv_sel2_rowlow = 1;
    common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
    common.in2_pad_cumv_sel2_collow = 1;
    common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
    common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
    common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
    common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
    common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

    // pointers
    d_in2_sub_cumh = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_sub_cumh[0] = (fp*) malloc(common.allPoints * common.in2_sub_cumh_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_sub_cumh[i] = d_in2_sub_cumh[i-1] + common.in2_sub_cumh_elem;
    }

    //====================================================================================================
    // 	SELECTION
    //====================================================================================================

    // common
    common.in2_sub_cumh_sel_rowlow = 1;
    common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
    common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
    common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
    common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
    common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
    common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
    common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

    // pointers
    d_in2_sub_cumh_sel = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_sub_cumh_sel[0] = (fp*) malloc(common.allPoints * common.in2_sub_cumh_sel_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_sub_cumh_sel[i] = d_in2_sub_cumh_sel[i-1] + common.in2_sub_cumh_sel_elem;
    }

    //====================================================================================================
    //	SELECTION 2, SUBTRACTION
    //====================================================================================================

    // common
    common.in2_sub_cumh_sel2_rowlow = 1;
    common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
    common.in2_sub_cumh_sel2_collow = 1;
    common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
    common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
    common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
    common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
    common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

    // pointers
    d_in2_sub2 = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_sub2[0] = (fp*) malloc(common.allPoints * common.in2_sub2_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_sub2[i] = d_in2_sub2[i-1] + common.in2_sub2_elem;
    }

    //======================================================================================================================================================
    //	CUMULATIVE SUM 2
    //======================================================================================================================================================

    //====================================================================================================
    //	MULTIPLICATION
    //====================================================================================================

    // common
    common.in2_sqr_rows = common.in2_rows;
    common.in2_sqr_cols = common.in2_cols;
    common.in2_sqr_elem = common.in2_elem;
    common.in2_sqr_mem = common.in2_mem;

    // pointers
    d_in2_sqr = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_sqr[0] = (fp*) malloc(common.allPoints * common.in2_sqr_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_sqr[i] = d_in2_sqr[i-1] + common.in2_elem;
    }

    //====================================================================================================
    //	SELECTION 2, SUBTRACTION
    //====================================================================================================

    // common
    common.in2_sqr_sub2_rows = common.in2_sub2_rows;
    common.in2_sqr_sub2_cols = common.in2_sub2_cols;
    common.in2_sqr_sub2_elem = common.in2_sub2_elem;
    common.in2_sqr_sub2_mem = common.in2_sub2_mem;

    // pointers
    d_in2_sqr_sub2 = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in2_sqr_sub2[0] = (fp*) malloc(common.allPoints * common.in2_sqr_sub2_mem);
    for(i=1; i<common.allPoints; i++){
        d_in2_sqr_sub2[i] = d_in2_sqr_sub2[i-1] + common.in2_sub2_elem;
    }

    //======================================================================================================================================================
    //	FINAL
    //======================================================================================================================================================

    // common
    common.in_sqr_rows = common.in_rows;
    common.in_sqr_cols = common.in_cols;
    common.in_sqr_elem = common.in_elem;
    common.in_sqr_mem = common.in_mem;

    // pointers
    d_in_sqr = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_in_sqr[0] = (fp*) malloc(common.allPoints * common.in_sqr_mem);
    for(i=1; i<common.allPoints; i++){
        d_in_sqr[i] = d_in_sqr[i-1] + common.in_sqr_elem;
    }

    //======================================================================================================================================================
    //	TEMPLATE MASK CREATE
    //======================================================================================================================================================

    // common
    common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
    common.tMask_cols = common.tMask_rows;
    common.tMask_elem = common.tMask_rows * common.tMask_cols;
    common.tMask_mem = sizeof(float) * common.tMask_elem;

    // pointers
    d_tMask = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_tMask[0] = (fp*) malloc(common.allPoints * common.tMask_mem);
    for(i=1; i<common.allPoints; i++){
        d_tMask[i] = d_tMask[i-1] + common.tMask_elem;
    }

    //======================================================================================================================================================
    //	POINT MASK INITIALIZE
    //======================================================================================================================================================

    // common
    common.mask_rows = common.maxMove;
    common.mask_cols = common.mask_rows;
    common.mask_elem = common.mask_rows * common.mask_cols;
    common.mask_mem = sizeof(float) * common.mask_elem;

    //======================================================================================================================================================
    //	MASK CONVOLUTION
    //======================================================================================================================================================

    // common
    common.mask_conv_rows = common.tMask_rows;												// number of rows in I
    common.mask_conv_cols = common.tMask_cols;												// number of columns in I
    common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
    common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
    common.mask_conv_ioffset = (common.mask_rows-1)/2;
    if((common.mask_rows-1) % 2 > 0.5){
        common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
    }
    common.mask_conv_joffset = (common.mask_cols-1)/2;
    if((common.mask_cols-1) % 2 > 0.5){
        common.mask_conv_joffset = common.mask_conv_joffset + 1;
    }

    // pointers
    d_mask_conv = (fp**) malloc(common.allPoints * sizeof(fp*));
    d_mask_conv[0] = (fp*) malloc(common.allPoints * common.mask_conv_mem);
    for(i=1; i<common.allPoints; i++){
        d_mask_conv[i] = d_mask_conv[i-1] + common.mask_conv_elem;
    }

    //====================================================================================================
    //	PRINT FRAME PROGRESS START
    //====================================================================================================

    printf("frame progress: ");
    fflush(NULL);

    //====================================================================================================
    //	LAUNCH
    //====================================================================================================

    for(common_change.frame_no=0; common_change.frame_no<frames_processed; common_change.frame_no++){

        // Extract a cropped version of the first frame from the video file
        frame = get_frame(	frames,						// pointer to video file
                                        common_change.frame_no,				// number of frame that needs to be returned
                                        0,								// cropped?
                                        0,								// scaled?
                                        1);							// converted

        // copy frame to GPU memory
        #pragma acc update target(common_change.frame[0:common.frame_elem])
        cudaMemcpyToSymbol(d_common_change, &common_change, sizeof(params_common_change));

        // launch GPU kernel
        kernel();

        // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
        free(frame);

        // print frame progress
        printf("%d ", common_change.frame_no);
        fflush(NULL);

    }

    //====================================================================================================
    //	PRINT FRAME PROGRESS END
    //====================================================================================================

    printf("\n");
    fflush(NULL);

} /* end acc data */

    //======================================================================================================================================================
    //	DEALLOCATION
    //======================================================================================================================================================

    //====================================================================================================
    //	COMMON
    //====================================================================================================

    // frame
    free(common_change.frame);

    // endo points
    free(common.endoRow);
    free(common.endoCol);
    free(common.tEndoRowLoc);
    free(common.tEndoColLoc);

    free(common.endoT);

    // epi points
    free(common.epiRow);
    free(common.epiCol);
    free(common.tEpiRowLoc);
    free(common.tEpiColLoc);

    free(common.epiT);

    //====================================================================================================
    //	POINTERS
    //====================================================================================================

    free(d_in2[0]);                 free(d_in2);
    free(d_conv[0]);                free(d_conv);
    free(d_in2_pad_cumv[0]);        free(d_in2_pad_cumv);
    free(d_in2_pad_cumv_sel[0]);    free(d_in2_pad_cumv_sel);
    free(d_in2_sub_cumh[0]);        free(d_in2_sub_cumh);
    free(d_in2_sub_cumh_sel[0]);    free(d_in2_sub_cumh_sel);
    free(d_in2_sub2[0]);            free(d_in2_sub2);
    free(d_in2_sqr[0]);             free(d_in2_sqr);
    free(d_in2_sqr_sub2[0]);        free(d_in2_sqr_sub2);
    free(d_in_sqr[0]);              free(d_in_sqr);
    free(d_tMask[0]);               free(d_tMask);
    free(d_mask_conv[0]);           free(d_mask_conv);

}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
