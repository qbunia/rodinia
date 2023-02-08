//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	KERNEL FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

void kernel(public_struct __public, private_struct __private) {

  //======================================================================================================================================================
  //	COMMON VARIABLES
  //======================================================================================================================================================

  int ei_new;
  fp *d_in;
  int rot_row;
  int rot_col;
  int in2_rowlow;
  int in2_collow;
  int jp1;
  int ja1, ja2;
  int ip1;
  int ia1, ia2;
  int ja, jb;
  int ia, ib;
  fp s;
  int i;
  int j;
  int row;
  int col;
  int ori_row;
  int ori_col;
  int position;
  fp sum;
  int pos_ori;
  fp temp;
  fp temp2;
  int cent;
  int tMask_row;
  int tMask_col;
  fp fin_max_val = 0;
  int fin_max_coo = 0;
  int largest_row;
  int largest_col;
  int offset_row;
  int offset_col;
  fp in_final_sum;
  fp in_sqr_final_sum;
  fp mean;
  fp mean_sqr;
  fp variance;
  fp deviation;
  fp denomT;
  int pointer;
  int ori_pointer;
  int loc_pointer;

  //======================================================================================================================================================
  //	GENERATE TEMPLATE
  //======================================================================================================================================================

  // generate templates based on the first frame only
  if (__public.frame_no == 0) {

    // update temporary row/col coordinates
    pointer = __private.point_no * __public.frames + __public.frame_no;
    __private.d_tRowLoc[pointer] = __private.d_Row[__private.point_no];
    __private.d_tColLoc[pointer] = __private.d_Col[__private.point_no];

    // pointers to: current frame, template for current point
    d_in = &__private.d_T[__private.in_pointer];

    // update template, limit the number of working threads to the size of
    // template
    for (col = 0; col < __public.in_mod_cols; col++) {
      for (row = 0; row < __public.in_mod_rows; row++) {

        // figure out row/col location in corresponding new template area in
        // image and give to every thread (get top left corner and progress down
        // and right)
        ori_row = __private.d_Row[__private.point_no] - 25 + row - 1;
        ori_col = __private.d_Col[__private.point_no] - 25 + col - 1;
        ori_pointer = ori_col * __public.frame_rows + ori_row;

        // update template
        d_in[col * __public.in_mod_rows + row] = __public.d_frame[ori_pointer];
      }
    }
  }

  //======================================================================================================================================================
  //	PROCESS POINTS
  //======================================================================================================================================================

  // process points in all frames except for the first one
  if (__public.frame_no != 0) {

    //====================================================================================================
    //	INPUTS
    //====================================================================================================

    //==================================================
    //	1) SETUP POINTER TO POINT TO CURRENT FRAME FROM BATCH
    //	2) SELECT INPUT 2 (SAMPLE AROUND POINT) FROM FRAME
    // SAVE IN d_in2 (NOT LINEAR IN MEMORY, SO NEED TO SAVE OUTPUT FOR LATER
    // EASY
    // USE)
    // 3) SQUARE INPUT 2
    // SAVE IN d_in2_sqr
    //==================================================

    // pointers and variables
    in2_rowlow = __private.d_Row[__private.point_no] - __public.sSize; // (1 to n+1)
    in2_collow = __private.d_Col[__private.point_no] - __public.sSize;

    // work
    for (col = 0; col < __public.in2_cols; col++) {
      for (row = 0; row < __public.in2_rows; row++) {

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + in2_rowlow - 1;
        ori_col = col + in2_collow - 1;
        temp = __public.d_frame[ori_col * __public.frame_rows + ori_row];
        __private.d_in2[col * __public.in2_rows + row] = temp;
        __private.d_in2_sqr[col * __public.in2_rows + row] = temp * temp;
      }
    }

    //==================================================
    //	1) GET POINTER TO INPUT 1 (TEMPLATE FOR THIS POINT) IN TEMPLATE ARRAY
    //(LINEAR IN MEMORY, SO DONT NEED TO SAVE, JUST GET POINTER) 	2)
    // ROTATE INPUT
    // 1 SAVE IN d_in_mod 	3) SQUARE INPUT 1
    // SAVE IN d_in_sqr
    //==================================================

    // variables
    d_in = &__private.d_T[__private.in_pointer];

    // work
    for (col = 0; col < __public.in_mod_cols; col++) {
      for (row = 0; row < __public.in_mod_rows; row++) {

        // rotated coordinates
        rot_row = (__public.in_mod_rows - 1) - row;
        rot_col = (__public.in_mod_rows - 1) - col;
        pointer = rot_col * __public.in_mod_rows + rot_row;

        // execution
        temp = d_in[pointer];
        __private.d_in_mod[col * __public.in_mod_rows + row] = temp;
        __private.d_in_sqr[pointer] = temp * temp;
      }
    }

    //==================================================
    //	1) GET SUM OF INPUT 1
    //	2) GET SUM OF INPUT 1 SQUARED
    //==================================================

    in_final_sum = 0;
    for (i = 0; i < __public.in_mod_elem; i++) {
      in_final_sum = in_final_sum + d_in[i];
    }

    in_sqr_final_sum = 0;
    for (i = 0; i < __public.in_mod_elem; i++) {
      in_sqr_final_sum = in_sqr_final_sum + __private.d_in_sqr[i];
    }

    //==================================================
    //	3) DO STATISTICAL CALCULATIONS
    //	4) GET DENOMINATOR T
    //==================================================

    mean = in_final_sum /
           __public.in_mod_elem; // gets mean (average) value of element in ROI
    mean_sqr = mean * mean;
    variance = (in_sqr_final_sum / __public.in_mod_elem) -
               mean_sqr;        // gets variance of ROI
    deviation = sqrt(variance); // gets standard deviation of ROI

    denomT = sqrt((fp)(__public.in_mod_elem - 1)) * deviation;

    //====================================================================================================
    //	1) CONVOLVE INPUT 2 WITH ROTATED INPUT 1
    // SAVE IN d_conv
    //====================================================================================================

    // work
    for (col = 1; col <= __public.conv_cols; col++) {

      // column setup
      j = col + __public.joffset;
      jp1 = j + 1;
      if (__public.in2_cols < jp1) {
        ja1 = jp1 - __public.in2_cols;
      } else {
        ja1 = 1;
      }
      if (__public.in_mod_cols < j) {
        ja2 = __public.in_mod_cols;
      } else {
        ja2 = j;
      }

      for (row = 1; row <= __public.conv_rows; row++) {

        // row range setup
        i = row + __public.ioffset;
        ip1 = i + 1;

        if (__public.in2_rows < ip1) {
          ia1 = ip1 - __public.in2_rows;
        } else {
          ia1 = 1;
        }
        if (__public.in_mod_rows < i) {
          ia2 = __public.in_mod_rows;
        } else {
          ia2 = i;
        }

        s = 0;

        // getting data
        for (ja = ja1; ja <= ja2; ja++) {
          jb = jp1 - ja;
          for (ia = ia1; ia <= ia2; ia++) {
            ib = ip1 - ia;
            s = s + __private.d_in_mod[__public.in_mod_rows * (ja - 1) + ia - 1] *
                        __private.d_in2[__public.in2_rows * (jb - 1) + ib - 1];
          }
        }

        __private.d_conv[(col - 1) * __public.conv_rows + (row - 1)] = s;
      }
    }
    //====================================================================================================
    //	LOCAL SUM 1
    //====================================================================================================

    //==================================================
    //	1) PADD ARRAY
    // SAVE IN d_in2_pad
    //==================================================

    // work
    for (col = 0; col < __public.in2_pad_cols; col++) {
      for (row = 0; row < __public.in2_pad_rows; row++) {

        // execution
        if (row > (__public.in2_pad_add_rows -
                   1) && // do if has numbers in original array
            row < (__public.in2_pad_add_rows + __public.in2_rows) &&
            col > (__public.in2_pad_add_cols - 1) &&
            col < (__public.in2_pad_add_cols + __public.in2_cols)) {
          ori_row = row - __public.in2_pad_add_rows;
          ori_col = col - __public.in2_pad_add_cols;
          __private.d_in2_pad[col * __public.in2_pad_rows + row] =
              __private.d_in2[ori_col * __public.in2_rows + ori_row];
        } else { // do if otherwise
          __private.d_in2_pad[col * __public.in2_pad_rows + row] = 0;
        }
      }
    }

    //==================================================
    //	1) GET VERTICAL CUMULATIVE SUM SAVE IN d_in2_pad
    //==================================================

    for (ei_new = 0; ei_new < __public.in2_pad_cols; ei_new++) {

      // figure out column position
      pos_ori = ei_new * __public.in2_pad_rows;

      // loop through all rows
      sum = 0;
      for (position = pos_ori; position < pos_ori + __public.in2_pad_rows;
           position = position + 1) {
        __private.d_in2_pad[position] = __private.d_in2_pad[position] + sum;
        sum = __private.d_in2_pad[position];
      }
    }

    //==================================================
    //	1) MAKE 1st SELECTION FROM VERTICAL CUMULATIVE SUM
    //	2) MAKE 2nd SELECTION FROM VERTICAL CUMULATIVE SUM
    //	3) SUBTRACT THE TWO SELECTIONS SAVE IN d_in2_sub
    //==================================================

    // work
    for (col = 0; col < __public.in2_sub_cols; col++) {
      for (row = 0; row < __public.in2_sub_rows; row++) {

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_pad_cumv_sel_rowlow - 1;
        ori_col = col + __public.in2_pad_cumv_sel_collow - 1;
        temp = __private.d_in2_pad[ori_col * __public.in2_pad_rows + ori_row];

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_pad_cumv_sel2_rowlow - 1;
        ori_col = col + __public.in2_pad_cumv_sel2_collow - 1;
        temp2 = __private.d_in2_pad[ori_col * __public.in2_pad_rows + ori_row];

        // subtraction
        __private.d_in2_sub[col * __public.in2_sub_rows + row] = temp - temp2;
      }
    }

    //==================================================
    //	1) GET HORIZONTAL CUMULATIVE SUM
    // SAVE IN d_in2_sub
    //==================================================

    for (ei_new = 0; ei_new < __public.in2_sub_rows; ei_new++) {

      // figure out row position
      pos_ori = ei_new;

      // loop through all rows
      sum = 0;
      for (position = pos_ori; position < pos_ori + __public.in2_sub_elem;
           position = position + __public.in2_sub_rows) {
        __private.d_in2_sub[position] = __private.d_in2_sub[position] + sum;
        sum = __private.d_in2_sub[position];
      }
    }

    //==================================================
    //	1) MAKE 1st SELECTION FROM HORIZONTAL CUMULATIVE SUM
    //	2) MAKE 2nd SELECTION FROM HORIZONTAL CUMULATIVE SUM
    //	3) SUBTRACT THE TWO SELECTIONS TO GET LOCAL SUM 1
    //	4) GET CUMULATIVE SUM 1 SQUARED SAVE IN d_in2_sub2_sqr 	5) GET NUMERATOR
    // SAVE IN d_conv
    //==================================================

    // work
    for (col = 0; col < __public.in2_sub2_sqr_cols; col++) {
      for (row = 0; row < __public.in2_sub2_sqr_rows; row++) {

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_sub_cumh_sel_rowlow - 1;
        ori_col = col + __public.in2_sub_cumh_sel_collow - 1;
        temp = __private.d_in2_sub[ori_col * __public.in2_sub_rows + ori_row];

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_sub_cumh_sel2_rowlow - 1;
        ori_col = col + __public.in2_sub_cumh_sel2_collow - 1;
        temp2 = __private.d_in2_sub[ori_col * __public.in2_sub_rows + ori_row];

        // subtraction
        temp2 = temp - temp2;

        // squaring
        __private.d_in2_sub2_sqr[col * __public.in2_sub2_sqr_rows + row] =
            temp2 * temp2;

        // numerator
        __private.d_conv[col * __public.in2_sub2_sqr_rows + row] =
            __private.d_conv[col * __public.in2_sub2_sqr_rows + row] -
            temp2 * in_final_sum / __public.in_mod_elem;
      }
    }

    //====================================================================================================
    //	LOCAL SUM 2
    //====================================================================================================

    //==================================================
    //	1) PAD ARRAY
    // SAVE IN d_in2_pad
    //==================================================

    // work
    for (col = 0; col < __public.in2_pad_cols; col++) {
      for (row = 0; row < __public.in2_pad_rows; row++) {

        // execution
        if (row > (__public.in2_pad_add_rows -
                   1) && // do if has numbers in original array
            row < (__public.in2_pad_add_rows + __public.in2_rows) &&
            col > (__public.in2_pad_add_cols - 1) &&
            col < (__public.in2_pad_add_cols + __public.in2_cols)) {
          ori_row = row - __public.in2_pad_add_rows;
          ori_col = col - __public.in2_pad_add_cols;
          __private.d_in2_pad[col * __public.in2_pad_rows + row] =
              __private.d_in2_sqr[ori_col * __public.in2_rows + ori_row];
        } else { // do if otherwise
          __private.d_in2_pad[col * __public.in2_pad_rows + row] = 0;
        }
      }
    }

    //==================================================
    //	2) GET VERTICAL CUMULATIVE SUM SAVE IN d_in2_pad
    //==================================================

    // work
    for (ei_new = 0; ei_new < __public.in2_pad_cols; ei_new++) {

      // figure out column position
      pos_ori = ei_new * __public.in2_pad_rows;

      // loop through all rows
      sum = 0;
      for (position = pos_ori; position < pos_ori + __public.in2_pad_rows;
           position = position + 1) {
        __private.d_in2_pad[position] = __private.d_in2_pad[position] + sum;
        sum = __private.d_in2_pad[position];
      }
    }

    //==================================================
    //	1) MAKE 1st SELECTION FROM VERTICAL CUMULATIVE SUM
    //	2) MAKE 2nd SELECTION FROM VERTICAL CUMULATIVE SUM
    //	3) SUBTRACT THE TWO SELECTIONS SAVE IN d_in2_sub
    //==================================================

    // work
    for (col = 0; col < __public.in2_sub_cols; col++) {
      for (row = 0; row < __public.in2_sub_rows; row++) {

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_pad_cumv_sel_rowlow - 1;
        ori_col = col + __public.in2_pad_cumv_sel_collow - 1;
        temp = __private.d_in2_pad[ori_col * __public.in2_pad_rows + ori_row];

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_pad_cumv_sel2_rowlow - 1;
        ori_col = col + __public.in2_pad_cumv_sel2_collow - 1;
        temp2 = __private.d_in2_pad[ori_col * __public.in2_pad_rows + ori_row];

        // subtract
        __private.d_in2_sub[col * __public.in2_sub_rows + row] = temp - temp2;
      }
    }

    //==================================================
    //	1) GET HORIZONTAL CUMULATIVE SUM
    // SAVE IN d_in2_sub
    //==================================================

    for (ei_new = 0; ei_new < __public.in2_sub_rows; ei_new++) {

      // figure out row position
      pos_ori = ei_new;

      // loop through all rows
      sum = 0;
      for (position = pos_ori; position < pos_ori + __public.in2_sub_elem;
           position = position + __public.in2_sub_rows) {
        __private.d_in2_sub[position] = __private.d_in2_sub[position] + sum;
        sum = __private.d_in2_sub[position];
      }
    }

    //==================================================
    //	1) MAKE 1st SELECTION FROM HORIZONTAL CUMULATIVE SUM
    //	2) MAKE 2nd SELECTION FROM HORIZONTAL CUMULATIVE SUM
    //	3) SUBTRACT THE TWO SELECTIONS TO GET LOCAL SUM 2
    //	4) GET DIFFERENTIAL LOCAL SUM
    //	5) GET DENOMINATOR A
    //	6) GET DENOMINATOR
    //	7) DIVIDE NUMBERATOR BY DENOMINATOR TO GET CORRELATION	SAVE IN d_conv
    //==================================================

    // work
    for (col = 0; col < __public.conv_cols; col++) {
      for (row = 0; row < __public.conv_rows; row++) {

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_sub_cumh_sel_rowlow - 1;
        ori_col = col + __public.in2_sub_cumh_sel_collow - 1;
        temp = __private.d_in2_sub[ori_col * __public.in2_sub_rows + ori_row];

        // figure out corresponding location in old matrix and copy values to
        // new matrix
        ori_row = row + __public.in2_sub_cumh_sel2_rowlow - 1;
        ori_col = col + __public.in2_sub_cumh_sel2_collow - 1;
        temp2 = __private.d_in2_sub[ori_col * __public.in2_sub_rows + ori_row];

        // subtract
        temp2 = temp - temp2;

        // diff_local_sums
        temp2 = temp2 - (__private.d_in2_sub2_sqr[col * __public.conv_rows + row] /
                         __public.in_mod_elem);

        // denominator A
        if (temp2 < 0) {
          temp2 = 0;
        }
        temp2 = sqrt(temp2);

        // denominator
        temp2 = denomT * temp2;

        // correlation
        __private.d_conv[col * __public.conv_rows + row] =
            __private.d_conv[col * __public.conv_rows + row] / temp2;
      }
    }

    //====================================================================================================
    //	TEMPLATE MASK CREATE
    //====================================================================================================

    // parameters
    cent = __public.sSize + __public.tSize + 1;
    pointer = __public.frame_no - 1 + __private.point_no * __public.frames;
    tMask_row =
        cent + __private.d_tRowLoc[pointer] - __private.d_Row[__private.point_no] - 1;
    tMask_col =
        cent + __private.d_tColLoc[pointer] - __private.d_Col[__private.point_no] - 1;

    // work
    for (ei_new = 0; ei_new < __public.tMask_elem; ei_new++) {
      __private.d_tMask[ei_new] = 0;
    }
    __private.d_tMask[tMask_col * __public.tMask_rows + tMask_row] = 1;

    //====================================================================================================
    //	1) MASK CONVOLUTION
    //	2) MULTIPLICATION
    //====================================================================================================

    // work
    // for(col=1; col<=__public.conv_cols; col++){
    for (col = 1; col <= __public.mask_conv_cols; col++) {

      // col setup
      j = col + __public.mask_conv_joffset;
      jp1 = j + 1;
      if (__public.mask_cols < jp1) {
        ja1 = jp1 - __public.mask_cols;
      } else {
        ja1 = 1;
      }
      if (__public.tMask_cols < j) {
        ja2 = __public.tMask_cols;
      } else {
        ja2 = j;
      }

      // for(row=1; row<=__public.conv_rows; row++){
      for (row = 1; row <= __public.mask_conv_rows; row++) {

        // row setup
        i = row + __public.mask_conv_ioffset;
        ip1 = i + 1;

        if (__public.mask_rows < ip1) {
          ia1 = ip1 - __public.mask_rows;
        } else {
          ia1 = 1;
        }
        if (__public.tMask_rows < i) {
          ia2 = __public.tMask_rows;
        } else {
          ia2 = i;
        }

        s = 0;

        // get data
        for (ja = ja1; ja <= ja2; ja++) {
          jb = jp1 - ja;
          for (ia = ia1; ia <= ia2; ia++) {
            ib = ip1 - ia;
            s = s + __private.d_tMask[__public.tMask_rows * (ja - 1) + ia - 1] * 1;
          }
        }

        __private.d_mask_conv[(col - 1) * __public.conv_rows + (row - 1)] =
            __private.d_conv[(col - 1) * __public.conv_rows + (row - 1)] * s;
      }
    }

    //====================================================================================================
    //	MAXIMUM VALUE
    //====================================================================================================

    //==================================================
    //	SEARCH
    //==================================================

    fin_max_val = 0;
    fin_max_coo = 0;
    for (i = 0; i < __public.mask_conv_elem; i++) {
      if (__private.d_mask_conv[i] > fin_max_val) {
        fin_max_val = __private.d_mask_conv[i];
        fin_max_coo = i;
      }
    }

    //==================================================
    //	OFFSET
    //==================================================

    // convert coordinate to row/col form
    largest_row = (fin_max_coo + 1) % __public.mask_conv_rows - 1; // (0-n) row
    largest_col = (fin_max_coo + 1) / __public.mask_conv_rows;     // (0-n) column
    if ((fin_max_coo + 1) % __public.mask_conv_rows == 0) {
      largest_row = __public.mask_conv_rows - 1;
      largest_col = largest_col - 1;
    }

    // calculate offset
    largest_row = largest_row + 1; // compensate to match MATLAB format (1-n)
    largest_col = largest_col + 1; // compensate to match MATLAB format (1-n)
    offset_row =
        largest_row - __public.in_mod_rows - (__public.sSize - __public.tSize);
    offset_col =
        largest_col - __public.in_mod_cols - (__public.sSize - __public.tSize);
    pointer = __private.point_no * __public.frames + __public.frame_no;
    __private.d_tRowLoc[pointer] = __private.d_Row[__private.point_no] + offset_row;
    __private.d_tColLoc[pointer] = __private.d_Col[__private.point_no] + offset_col;
  }

  //======================================================================================================================================================
  //	COORDINATE AND TEMPLATE UPDATE
  //======================================================================================================================================================

  // if the last frame in the bath, update template
  if (__public.frame_no != 0 && (__public.frame_no) % 10 == 0) {

    // update coordinate
    loc_pointer = __private.point_no * __public.frames + __public.frame_no;
    __private.d_Row[__private.point_no] = __private.d_tRowLoc[loc_pointer];
    __private.d_Col[__private.point_no] = __private.d_tColLoc[loc_pointer];

    // update template, limit the number of working threads to the size of
    // template
    for (col = 0; col < __public.in_mod_cols; col++) {
      for (row = 0; row < __public.in_mod_rows; row++) {

        // figure out row/col location in corresponding new template area in
        // image and give to every thread (get top left corner and progress down
        // and right)
        ori_row = __private.d_Row[__private.point_no] - 25 + row - 1;
        ori_col = __private.d_Col[__private.point_no] - 25 + col - 1;
        ori_pointer = ori_col * __public.frame_rows + ori_row;

        // update template
        d_in[col * __public.in_mod_rows + row] =
            __public.alpha * d_in[col * __public.in_mod_rows + row] +
            (1.00 - __public.alpha) * __public.d_frame[ori_pointer];
      }
    }
  }
}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	END OF FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
