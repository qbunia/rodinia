//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "AVI/avilib.h"
#include "AVI/avimod.h"
#include <omp.h>

#include "define.h"
#include "kernel.c"

#define NUM_TEAMS 256
#define NUM_THREADS 1024

//===============================================================================================================================================================================================================200
//	WRITE DATA FUNCTION
//===============================================================================================================================================================================================================200

void write_data(char *filename, int frameNo, int frames_processed,
                int endoPoints, int *input_a, int *input_b, int epiPoints,
                int *input_2a, int *input_2b) {

  //================================================================================80
  //	VARIABLES
  //================================================================================80

  FILE *fid;
  int i, j;

  //================================================================================80
  //	OPEN FILE FOR READING
  //================================================================================80

  fid = fopen(filename, "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return;
  }

  //================================================================================80
  //	WRITE VALUES TO THE FILE
  //================================================================================80
  fprintf(fid, "Total AVI Frames: %d\n", frameNo);
  fprintf(fid, "Frames Processed: %d\n", frames_processed);
  fprintf(fid, "endoPoints: %d\n", endoPoints);
  fprintf(fid, "epiPoints: %d", epiPoints);
  for (j = 0; j < frames_processed; j++) {
    fprintf(fid, "\n---Frame %d---", j);
    fprintf(fid, "\n--endo--\n");
    for (i = 0; i < endoPoints; i++) {
      fprintf(fid, "%d\t", input_a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < endoPoints; i++) {
      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
      fprintf(fid, "%d\t", input_b[j + i * frameNo]);
    }
    fprintf(fid, "\n--epi--\n");
    for (i = 0; i < epiPoints; i++) {
      // if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < epiPoints; i++) {
      // if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2b[j + i * frameNo]);
    }
  }
  // 	================================================================================80
  //		CLOSE FILE
  //	================================================================================80

  fclose(fid);
}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

int main(int argc, char *argv[]) {

  //======================================================================================================================================================
  //	VARIABLES
  //======================================================================================================================================================

  // counters
  int i;
  int frames_processed;

  // parameters
  public_struct public;
  private_struct private[ALL_POINTS];

  //======================================================================================================================================================
  // 	FRAMES
  //======================================================================================================================================================

  if (argc != 4) {
    printf("ERROR: usage: heartwall <inputfile> <num of frames> <num of "
           "threads>\n");
    exit(1);
  }

  char *video_file_name;
  video_file_name = argv[1];

  avi_t *d_frames =
      (avi_t *)AVI_open_input_file(video_file_name, 1); // added casting
  if (d_frames == NULL) {
    AVI_print_error((char *)"Error with AVI_open_input_file");
    return -1;
  }

  int host_id = omp_get_initial_device();
  int device_id = omp_get_default_device();

  public.d_frames = d_frames;
  public.frames = AVI_video_frames(public.d_frames);
  public.frame_rows = AVI_video_height(public.d_frames);
  public.frame_cols = AVI_video_width(public.d_frames);
  public.frame_elem = public.frame_rows * public.frame_cols;
  public.frame_mem = sizeof(fp) * public.frame_elem;

  //======================================================================================================================================================
  // 	CHECK INPUT ARGUMENTS
  //======================================================================================================================================================

  frames_processed = atoi(argv[2]);
  if (frames_processed < 0 || frames_processed > public.frames) {
    printf("ERROR: %d is an incorrect number of frames specified, select in "
           "the range of 0-%d\n",
           frames_processed, public.frames);
    return 0;
  }

  int omp_num_threads;
  omp_num_threads = atoi(argv[3]);
  if (omp_num_threads <= 0) {
    printf("num of threads must be a positive integer");
    return 0;
  }

  printf("num of threads: %d\n", omp_num_threads);

  //======================================================================================================================================================
  //	INPUTS
  //======================================================================================================================================================

  //====================================================================================================
  //	ENDO POINTS
  //====================================================================================================

  public.endoPoints = ENDO_POINTS;
  public.d_endo_mem = sizeof(int) * public.endoPoints;

  public.h_endoRow = (int *)malloc(public.d_endo_mem);
  public.h_endoRow[0] = 369;
  public.h_endoRow[1] = 400;
  public.h_endoRow[2] = 429;
  public.h_endoRow[3] = 452;
  public.h_endoRow[4] = 476;
  public.h_endoRow[5] = 486;
  public.h_endoRow[6] = 479;
  public.h_endoRow[7] = 458;
  public.h_endoRow[8] = 433;
  public.h_endoRow[9] = 404;
  public.h_endoRow[10] = 374;
  public.h_endoRow[11] = 346;
  public.h_endoRow[12] = 318;
  public.h_endoRow[13] = 294;
  public.h_endoRow[14] = 277;
  public.h_endoRow[15] = 269;
  public.h_endoRow[16] = 275;
  public.h_endoRow[17] = 287;
  public.h_endoRow[18] = 311;
  public.h_endoRow[19] = 339;
  public.d_endoRow = omp_target_alloc(public.d_endo_mem, device_id);
  omp_target_memcpy(public.d_endoRow, public.h_endoRow, public.d_endo_mem, 0, 0,
                    device_id, host_id);

  public.h_endoCol = (int *)malloc(public.d_endo_mem);
  public.h_endoCol[0] = 408;
  public.h_endoCol[1] = 406;
  public.h_endoCol[2] = 397;
  public.h_endoCol[3] = 383;
  public.h_endoCol[4] = 354;
  public.h_endoCol[5] = 322;
  public.h_endoCol[6] = 294;
  public.h_endoCol[7] = 270;
  public.h_endoCol[8] = 250;
  public.h_endoCol[9] = 237;
  public.h_endoCol[10] = 235;
  public.h_endoCol[11] = 241;
  public.h_endoCol[12] = 254;
  public.h_endoCol[13] = 273;
  public.h_endoCol[14] = 300;
  public.h_endoCol[15] = 328;
  public.h_endoCol[16] = 356;
  public.h_endoCol[17] = 383;
  public.h_endoCol[18] = 401;
  public.h_endoCol[19] = 411;
  public.d_endoCol = omp_target_alloc(public.d_endo_mem, device_id);
  omp_target_memcpy(public.d_endoCol, public.h_endoCol, public.d_endo_mem, 0, 0,
                    device_id, host_id);

  public.h_tEndoRowLoc = (int *)malloc(public.d_endo_mem * public.frames);
  public.d_tEndoRowLoc =
      omp_target_alloc(public.d_endo_mem * public.frames, device_id);
  public.h_tEndoColLoc = (int *)malloc(public.d_endo_mem * public.frames);
  public.d_tEndoColLoc =
      omp_target_alloc(public.d_endo_mem * public.frames, device_id);

  //====================================================================================================
  //	EPI POINTS
  //====================================================================================================

  public.epiPoints = EPI_POINTS;
  public.d_epi_mem = sizeof(int) * public.epiPoints;

  public.h_epiRow = (int *)malloc(public.d_epi_mem);
  public.h_epiRow[0] = 390;
  public.h_epiRow[1] = 419;
  public.h_epiRow[2] = 448;
  public.h_epiRow[3] = 474;
  public.h_epiRow[4] = 501;
  public.h_epiRow[5] = 519;
  public.h_epiRow[6] = 535;
  public.h_epiRow[7] = 542;
  public.h_epiRow[8] = 543;
  public.h_epiRow[9] = 538;
  public.h_epiRow[10] = 528;
  public.h_epiRow[11] = 511;
  public.h_epiRow[12] = 491;
  public.h_epiRow[13] = 466;
  public.h_epiRow[14] = 438;
  public.h_epiRow[15] = 406;
  public.h_epiRow[16] = 376;
  public.h_epiRow[17] = 347;
  public.h_epiRow[18] = 318;
  public.h_epiRow[19] = 291;
  public.h_epiRow[20] = 275;
  public.h_epiRow[21] = 259;
  public.h_epiRow[22] = 256;
  public.h_epiRow[23] = 252;
  public.h_epiRow[24] = 252;
  public.h_epiRow[25] = 257;
  public.h_epiRow[26] = 266;
  public.h_epiRow[27] = 283;
  public.h_epiRow[28] = 305;
  public.h_epiRow[29] = 331;
  public.h_epiRow[30] = 360;
  public.d_epiRow = omp_target_alloc(public.d_epi_mem, device_id);
  omp_target_memcpy(public.d_epiRow, public.h_epiRow, public.d_epi_mem, 0, 0,
                    device_id, host_id);

  public.h_epiCol = (int *)malloc(public.d_epi_mem);
  public.h_epiCol[0] = 457;
  public.h_epiCol[1] = 454;
  public.h_epiCol[2] = 446;
  public.h_epiCol[3] = 431;
  public.h_epiCol[4] = 411;
  public.h_epiCol[5] = 388;
  public.h_epiCol[6] = 361;
  public.h_epiCol[7] = 331;
  public.h_epiCol[8] = 301;
  public.h_epiCol[9] = 273;
  public.h_epiCol[10] = 243;
  public.h_epiCol[11] = 218;
  public.h_epiCol[12] = 196;
  public.h_epiCol[13] = 178;
  public.h_epiCol[14] = 166;
  public.h_epiCol[15] = 157;
  public.h_epiCol[16] = 155;
  public.h_epiCol[17] = 165;
  public.h_epiCol[18] = 177;
  public.h_epiCol[19] = 197;
  public.h_epiCol[20] = 218;
  public.h_epiCol[21] = 248;
  public.h_epiCol[22] = 276;
  public.h_epiCol[23] = 304;
  public.h_epiCol[24] = 333;
  public.h_epiCol[25] = 361;
  public.h_epiCol[26] = 391;
  public.h_epiCol[27] = 415;
  public.h_epiCol[28] = 434;
  public.h_epiCol[29] = 448;
  public.h_epiCol[30] = 455;
  public.d_epiCol = omp_target_alloc(public.d_epi_mem, device_id);
  omp_target_memcpy(public.d_epiCol, public.h_epiCol, public.d_epi_mem, 0, 0,
                    device_id, host_id);

  public.h_tEpiRowLoc = (int *)malloc(public.d_epi_mem * public.frames);
  public.d_tEpiRowLoc =
      omp_target_alloc(public.d_epi_mem * public.frames, device_id);
  public.h_tEpiColLoc = (int *)malloc(public.d_epi_mem * public.frames);
  public.d_tEpiColLoc =
      omp_target_alloc(public.d_epi_mem * public.frames, device_id);

  //====================================================================================================
  //	ALL POINTS
  //====================================================================================================

  public.allPoints = ALL_POINTS;

  //======================================================================================================================================================
  //	CONSTANTS
  //======================================================================================================================================================

  public.tSize = 25;
  public.sSize = 40;
  public.maxMove = 10;
  public.alpha = 0.87;

  //======================================================================================================================================================
  //	SUMS
  //======================================================================================================================================================

  for (i = 0; i < public.allPoints; i++) {
    private[i].in_partial_sum =
        omp_target_alloc(sizeof(fp) * 2 * public.tSize + 1, device_id);
    private[i].in_sqr_partial_sum =
        omp_target_alloc(sizeof(fp) * 2 * public.tSize + 1, device_id);
    private[i].par_max_val = omp_target_alloc(
        sizeof(fp) * (2 * public.tSize + 2 * public.sSize + 1), device_id);
    private[i].par_max_coo = omp_target_alloc(
        sizeof(int) * (2 * public.tSize + 2 * public.sSize + 1), device_id);
  }

  //======================================================================================================================================================
  // 	INPUT 2 (SAMPLE AROUND POINT)
  //======================================================================================================================================================

  public.in2_rows = 2 * public.sSize + 1;
  public.in2_cols = 2 * public.sSize + 1;
  public.in2_elem = public.in2_rows * public.in2_cols;
  public.in2_mem = sizeof(fp) * public.in2_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_in2 = omp_target_alloc(public.in2_mem, device_id);
    private[i].d_in2_sqr = omp_target_alloc(public.in2_mem, device_id);
  }

  //======================================================================================================================================================
  // 	INPUT (POINT TEMPLATE)
  //======================================================================================================================================================

  public.in_mod_rows = public.tSize + 1 + public.tSize;
  public.in_mod_cols = public.in_mod_rows;
  public.in_mod_elem = public.in_mod_rows * public.in_mod_cols;
  public.in_mod_mem = sizeof(fp) * public.in_mod_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_in_mod = omp_target_alloc(public.in_mod_mem, device_id);
    private[i].d_in_sqr = omp_target_alloc(public.in_mod_mem, device_id);
  }

  //======================================================================================================================================================
  // 	ARRAY OF TEMPLATES FOR ALL POINTS
  //======================================================================================================================================================

  public.d_endoT =
      omp_target_alloc(public.in_mod_mem * public.endoPoints, device_id);
  public.d_epiT =
      omp_target_alloc(public.in_mod_mem * public.epiPoints, device_id);

  //======================================================================================================================================================
  // 	SETUP private POINTERS TO ROWS, COLS  AND TEMPLATE
  //======================================================================================================================================================

  for (i = 0; i < public.endoPoints; i++) {
    private[i].point_no = i;
    private[i].in_pointer = private[i].point_no * public.in_mod_elem;
    private[i].d_Row = public.d_endoRow;         // original row coordinates
    private[i].d_Col = public.d_endoCol;         // original col coordinates
    private[i].d_tRowLoc = public.d_tEndoRowLoc; // updated row coordinates
    private[i].d_tColLoc = public.d_tEndoColLoc; // updated row coordinates
    private[i].d_T = public.d_endoT;             // templates
  }

  for (i = public.endoPoints; i < public.allPoints; i++) {
    private[i].point_no = i - public.endoPoints;
    private[i].in_pointer = private[i].point_no * public.in_mod_elem;
    private[i].d_Row = public.d_epiRow;
    private[i].d_Col = public.d_epiCol;
    private[i].d_tRowLoc = public.d_tEpiRowLoc;
    private[i].d_tColLoc = public.d_tEpiColLoc;
    private[i].d_T = public.d_epiT;
  }

  //======================================================================================================================================================
  // 	CONVOLUTION
  //======================================================================================================================================================

  public.ioffset = 0;
  public.joffset = 0;
  public.conv_rows =
      public.in_mod_rows + public.in2_rows - 1; // number of rows in I
  public.conv_cols =
      public.in_mod_cols + public.in2_cols - 1; // number of columns in I
  public.conv_elem = public.conv_rows * public.conv_cols; // number of elements
  public.conv_mem = sizeof(fp) * public.conv_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_conv = omp_target_alloc(public.conv_mem, device_id);
  }

  //======================================================================================================================================================
  // 	CUMULATIVE SUM
  //======================================================================================================================================================

  //====================================================================================================
  //	PAD ARRAY
  //====================================================================================================
  //====================================================================================================
  //	VERTICAL CUMULATIVE SUM
  //====================================================================================================

  public.in2_pad_add_rows = public.in_mod_rows;
  public.in2_pad_add_cols = public.in_mod_cols;
  public.in2_pad_rows = public.in2_rows + 2 * public.in2_pad_add_rows;
  public.in2_pad_cols = public.in2_cols + 2 * public.in2_pad_add_cols;
  public.in2_pad_elem = public.in2_pad_rows * public.in2_pad_cols;
  public.in2_pad_mem = sizeof(fp) * public.in2_pad_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_in2_pad = omp_target_alloc(public.in2_pad_mem, device_id);
  }

  //====================================================================================================
  //	SELECTION, SELECTION 2, SUBTRACTION
  //====================================================================================================
  //====================================================================================================
  //	HORIZONTAL CUMULATIVE SUM
  //====================================================================================================

  public.in2_pad_cumv_sel_rowlow = 1 + public.in_mod_rows; // (1 to n+1)
  public.in2_pad_cumv_sel_rowhig = public.in2_pad_rows - 1;
  public.in2_pad_cumv_sel_collow = 1;
  public.in2_pad_cumv_sel_colhig = public.in2_pad_cols;
  public.in2_pad_cumv_sel2_rowlow = 1;
  public.in2_pad_cumv_sel2_rowhig =
      public.in2_pad_rows - public.in_mod_rows - 1;
  public.in2_pad_cumv_sel2_collow = 1;
  public.in2_pad_cumv_sel2_colhig = public.in2_pad_cols;
  public.in2_sub_rows =
      public.in2_pad_cumv_sel_rowhig - public.in2_pad_cumv_sel_rowlow + 1;
  public.in2_sub_cols =
      public.in2_pad_cumv_sel_colhig - public.in2_pad_cumv_sel_collow + 1;
  public.in2_sub_elem = public.in2_sub_rows * public.in2_sub_cols;
  public.in2_sub_mem = sizeof(fp) * public.in2_sub_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_in2_sub = omp_target_alloc(public.in2_sub_mem, device_id);
  }

  //====================================================================================================
  //	SELECTION, SELECTION 2, SUBTRACTION, SQUARE, NUMERATOR
  //====================================================================================================

  public.in2_sub_cumh_sel_rowlow = 1;
  public.in2_sub_cumh_sel_rowhig = public.in2_sub_rows;
  public.in2_sub_cumh_sel_collow = 1 + public.in_mod_cols;
  public.in2_sub_cumh_sel_colhig = public.in2_sub_cols - 1;
  public.in2_sub_cumh_sel2_rowlow = 1;
  public.in2_sub_cumh_sel2_rowhig = public.in2_sub_rows;
  public.in2_sub_cumh_sel2_collow = 1;
  public.in2_sub_cumh_sel2_colhig =
      public.in2_sub_cols - public.in_mod_cols - 1;
  public.in2_sub2_sqr_rows =
      public.in2_sub_cumh_sel_rowhig - public.in2_sub_cumh_sel_rowlow + 1;
  public.in2_sub2_sqr_cols =
      public.in2_sub_cumh_sel_colhig - public.in2_sub_cumh_sel_collow + 1;
  public.in2_sub2_sqr_elem =
      public.in2_sub2_sqr_rows * public.in2_sub2_sqr_cols;
  public.in2_sub2_sqr_mem = sizeof(fp) * public.in2_sub2_sqr_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_in2_sub2_sqr =
        omp_target_alloc(public.in2_sub2_sqr_mem, device_id);
  }

  //======================================================================================================================================================
  //	CUMULATIVE SUM 2
  //======================================================================================================================================================

  //====================================================================================================
  //	PAD ARRAY
  //====================================================================================================
  //====================================================================================================
  //	VERTICAL CUMULATIVE SUM
  //====================================================================================================

  //====================================================================================================
  //	SELECTION, SELECTION 2, SUBTRACTION
  //====================================================================================================
  //====================================================================================================
  //	HORIZONTAL CUMULATIVE SUM
  //====================================================================================================

  //====================================================================================================
  //	SELECTION, SELECTION 2, SUBTRACTION, DIFFERENTIAL LOCAL SUM, DENOMINATOR
  // A, DENOMINATOR, CORRELATION
  //====================================================================================================

  //======================================================================================================================================================
  //	TEMPLATE MASK CREATE
  //======================================================================================================================================================

  public.tMask_rows =
      public.in_mod_rows + (public.sSize + 1 + public.sSize) - 1;
  public.tMask_cols = public.tMask_rows;
  public.tMask_elem = public.tMask_rows * public.tMask_cols;
  public.tMask_mem = sizeof(fp) * public.tMask_elem;

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_tMask = omp_target_alloc(public.tMask_mem, device_id);
  }

  //======================================================================================================================================================
  //	POINT MASK INITIALIZE
  //======================================================================================================================================================

  public.mask_rows = public.maxMove;
  public.mask_cols = public.mask_rows;
  public.mask_elem = public.mask_rows * public.mask_cols;
  public.mask_mem = sizeof(fp) * public.mask_elem;

  //======================================================================================================================================================
  //	MASK CONVOLUTION
  //======================================================================================================================================================

  public.mask_conv_rows = public.tMask_rows; // number of rows in I
  public.mask_conv_cols = public.tMask_cols; // number of columns in I
  public.mask_conv_elem =
      public.mask_conv_rows * public.mask_conv_cols; // number of elements
  public.mask_conv_mem = sizeof(fp) * public.mask_conv_elem;
  public.mask_conv_ioffset = (public.mask_rows - 1) / 2;
  if ((public.mask_rows - 1) % 2 > 0.5) {
    public.mask_conv_ioffset = public.mask_conv_ioffset + 1;
  }
  public.mask_conv_joffset = (public.mask_cols - 1) / 2;
  if ((public.mask_cols - 1) % 2 > 0.5) {
    public.mask_conv_joffset = public.mask_conv_joffset + 1;
  }

  for (i = 0; i < public.allPoints; i++) {
    private[i].d_mask_conv = omp_target_alloc(public.mask_conv_mem, device_id);
  }

  //======================================================================================================================================================
  //	PRINT FRAME PROGRESS START
  //======================================================================================================================================================

  printf("frame progress: ");
  fflush(NULL);

  //======================================================================================================================================================
  //	KERNEL
  //======================================================================================================================================================

  public.d_frame = omp_target_alloc(public.frame_mem, device_id);

  for (public.frame_no = 0; public.frame_no < frames_processed;
       public.frame_no++) {

    //====================================================================================================
    //	GETTING FRAME
    //====================================================================================================

    // Extract a cropped version of the first frame from the video file
    float *host_frame =
        get_frame(public.d_frames, // pointer to video file
                  public.frame_no, // number of frame that needs to be returned
                  0,               // cropped?
                  0,               // scaled?
                  1);              // converted
    omp_target_memcpy(public.d_frame, host_frame, public.frame_mem, 0, 0,
                      device_id, host_id);

    //====================================================================================================
    //	PROCESSING
    //====================================================================================================
#pragma omp target data map(to : public, private[0 : ALL_POINTS])
    {
#pragma omp target teams distribute parallel for map(                          \
        to : public, private[0 : ALL_POINTS]) num_teams(NUM_TEAMS)             \
    num_threads(NUM_THREADS)
      for (i = 0; i < public.allPoints; i++) {
        kernel(public, private[i]);
      }
    }
    //====================================================================================================
    //	FREE MEMORY FOR FRAME
    //====================================================================================================

    // free frame after each loop iteration, since AVI library allocates
    // memory for every frame fetched
    free(host_frame);

    //====================================================================================================
    //	PRINT FRAME PROGRESS
    //====================================================================================================

    printf("%d ", public.frame_no);
    fflush(NULL);

    //======================================================================================================================================================
    //	PRINT FRAME PROGRESS END
    //======================================================================================================================================================

    printf("\n");
    fflush(NULL);
  }

  omp_target_memcpy(public.h_tEndoRowLoc, public.d_tEndoRowLoc,
                    public.d_endo_mem * public.frames, 0, 0, host_id,
                    device_id);
  omp_target_memcpy(public.h_tEndoColLoc, public.d_tEndoColLoc,
                    public.d_endo_mem * public.frames, 0, 0, host_id,
                    device_id);
  omp_target_memcpy(public.h_tEpiRowLoc, public.d_tEpiRowLoc,
                    public.d_epi_mem * public.frames, 0, 0, host_id, device_id);
  omp_target_memcpy(public.h_tEpiColLoc, public.d_tEpiColLoc,
                    public.d_epi_mem * public.frames, 0, 0, host_id, device_id);

  //==================================================50
  //	DUMP DATA TO FILE
  //==================================================50
  write_data("result.log", public.frames, frames_processed, public.endoPoints,
             public.h_tEndoRowLoc, public.h_tEndoColLoc, public.epiPoints,
             public.h_tEpiRowLoc, public.h_tEpiColLoc);

  //======================================================================================================================================================
  //	DEALLOCATION
  //======================================================================================================================================================

  //====================================================================================================
  //	COMMON
  //====================================================================================================

  omp_target_free(public.d_endoRow, device_id);
  omp_target_free(public.d_endoCol, device_id);
  omp_target_free(public.d_tEndoRowLoc, device_id);
  omp_target_free(public.d_tEndoColLoc, device_id);
  omp_target_free(public.d_endoT, device_id);

  free(public.h_endoRow);
  free(public.h_endoCol);
  free(public.h_tEndoRowLoc);
  free(public.h_tEndoColLoc);

  omp_target_free(public.d_epiRow, device_id);
  omp_target_free(public.d_epiCol, device_id);
  omp_target_free(public.d_tEpiRowLoc, device_id);
  omp_target_free(public.d_tEpiColLoc, device_id);
  omp_target_free(public.d_epiT, device_id);

  free(public.h_epiRow);
  free(public.h_epiCol);
  free(public.h_tEpiRowLoc);
  free(public.h_tEpiColLoc);

  omp_target_free(public.d_frame, device_id);

  //====================================================================================================
  //	POINTERS
  //====================================================================================================

  for (i = 0; i < public.allPoints; i++) {
    omp_target_free(private[i].in_partial_sum, device_id);
    omp_target_free(private[i].in_sqr_partial_sum, device_id);
    omp_target_free(private[i].par_max_val, device_id);
    omp_target_free(private[i].par_max_coo, device_id);

    omp_target_free(private[i].d_in2, device_id);
    omp_target_free(private[i].d_in2_sqr, device_id);

    omp_target_free(private[i].d_in_mod, device_id);
    omp_target_free(private[i].d_in_sqr, device_id);

    omp_target_free(private[i].d_conv, device_id);

    omp_target_free(private[i].d_in2_pad, device_id);

    omp_target_free(private[i].d_in2_sub, device_id);

    omp_target_free(private[i].d_in2_sub2_sqr, device_id);

    omp_target_free(private[i].d_tMask, device_id);
    omp_target_free(private[i].d_mask_conv, device_id);
  }
}

//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
//	END OF FILE
//========================================================================================================================================================================================================
//========================================================================================================================================================================================================
