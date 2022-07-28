//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	KERNEL FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

void kernel(){

	//======================================================================================================================================================
	//	COMMON VARIABLES
	//======================================================================================================================================================

	fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
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
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	fp largest_value_current = 0;
	fp largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	fp fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	fp in_final_sum = 0.f;
	fp in_sqr_final_sum = 0.f;
	float mean;
	float mean_sqr;
	float variance;
	float deviation;
	__shared__ float denomT;
	__shared__ float par_max_val[131];															// WATCH THIS !!! HARDCODED VALUE
	__shared__ int par_max_coo[131];															// WATCH THIS !!! HARDCODED VALUE
	int pointer;
	__shared__ float d_in_mod_temp[2601];
	int ori_pointer;
	int loc_pointer;
	int ei_new;

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	GENERATE TEMPLATE
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// generate templates based on the first frame only
	if(common_change.frame_no == 0){

		//======================================================================================================================================================
		// GET POINTER TO TEMPLATE FOR THE POINT
		//======================================================================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// pointers to: current template for current point
		d_in = &d_T[in_pointer[p]];

		//======================================================================================================================================================
		//	UPDATE ROW LOC AND COL LOC
		//======================================================================================================================================================

		// uptade temporary endo/epi row/col coordinates
		pointer = point_no[p]*common.no_frames+common_change.frame_no;
		tRowLoc[pointer] = Row[point_no[p]];
		tColLoc[pointer] = Col[point_no[p]];

		//======================================================================================================================================================
		//	CREATE TEMPLATES
		//======================================================================================================================================================

		// work
		for(ei_new=0; ei_new<common.in_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in_rows == 0){
				row = common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = Row[point_no[p]] - 25 + row - 1;
			ori_col = Col[point_no[p]] - 25 + col - 1;
			ori_pointer = ori_col*common.frame_rows+ori_row;

			// update template
			d_in[col*common.in_rows+row] = common_change.frame[ori_pointer];

		}

		}

	}

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	PROCESS POINTS
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// process points in all frames except for the first one
	if(common_change.frame_no != 0){

		//======================================================================================================================================================
		//	SELECTION
		//======================================================================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		in2_rowlow = Row[point_no[p]] - common.sSize;													// (1 to n+1)
		in2_collow = Col[point_no[p]] - common.sSize;

		// work
		for(ei_new=0; ei_new<common.in2_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_rows == 0){
				row = common.in2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_in2[p][ei_new] = common_change.frame[ori_col*common.frame_rows+ori_row];

		}

		}

		//======================================================================================================================================================
		//	CONVOLUTION
		//======================================================================================================================================================

		//====================================================================================================
		//	ROTATION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// variables
		d_in = &d_T[in_pointer];

		// work
		for(ei_new=0; ei_new<common.in_elem; ei_new++){

			// figure out row/col location in padded array
			row = (ei_new+1) % common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in_rows == 0){
				row = common.in_rows - 1;
				col = col-1;
			}
		
			// execution
			rot_row = (common.in_rows-1) - row;
			rot_col = (common.in_rows-1) - col;
			temp = d_in[rot_col*common.in_rows+rot_row];
			// TODO: private.d_in_mod[col*public.in_mod_rows+row] = temp;
			// TODO: private.d_in_sqr[pointer] = temp * temp;

		}

		}

		//====================================================================================================
		//	ACTUAL CONVOLUTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.conv_elem; ei_new++){

			// figure out row/col location in array
			ic = (ei_new+1) % common.conv_rows;												// (1-n)
			jc = (ei_new+1) / common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % common.conv_rows == 0){
				ic = common.conv_rows;
				jc = jc-1;
			}

			//
			j = jc + common.joffset;
			jp1 = j + 1;
			if(common.in2_cols < jp1){
				ja1 = jp1 - common.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(common.in_cols < j){
				ja2 = common.in_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + common.ioffset;
			ip1 = i + 1;
			
			if(common.in2_rows < ip1){
				ia1 = ip1 - common.in2_rows;
			}
			else{
				ia1 = 1;
			}
			if(common.in_rows < i){
				ia2 = common.in_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_in_mod_temp[common.in_rows*(ja-1)+ia-1] * d_in2[common.in2_rows*(jb-1)+ib-1];
				}
			}

			//d_conv[common.conv_rows*(jc-1)+ic-1] = s;
			d_conv[ei_new] = s;

		}

		}

		//======================================================================================================================================================
		//	CUMULATIVE SUM
		//======================================================================================================================================================

		//====================================================================================================
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//====================================================================================================
	
		//==================================================
		//	PADD ARRAY
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_pad_cumv_elem; ei_new++){

			// figure out row/col location in padded array
			row = (ei_new+1) % common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_pad_cumv_rows == 0){
				row = common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (common.in2_pad_add_rows+common.in2_rows) && 
				col > (common.in2_pad_add_cols-1) && 
				col < (common.in2_pad_add_cols+common.in2_cols)){
				ori_row = row - common.in2_pad_add_rows;
				ori_col = col - common.in2_pad_add_cols;
				d_in2_pad_cumv[p][ei_new] = d_in2[p][ori_col*common.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				d_in2_pad_cumv[p][ei_new] = 0;
			}

		}

		}


		//==================================================
		//	VERTICAL CUMULATIVE SUM
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		//work
		for(ei_new=0; ei_new<common.in2_pad_cumv_cols; ei_new++){

			// figure out column position
			pos_ori = ei_new*common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+common.in2_pad_cumv_rows; position = position + 1){
				d_in2_pad_cumv[p][position] = d_in2_pad_cumv[p][position] + sum;
				sum = d_in2_pad_cumv[p][position];
			}

		}

		}


		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_pad_cumv_sel_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_pad_cumv_sel_rows == 0){
				row = common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + common.in2_pad_cumv_sel_collow - 1;
			d_in2_pad_cumv_sel[p][ei_new] = d_in2_pad_cumv[p][ori_col*common.in2_pad_cumv_rows+ori_row];
		}

		}


		//====================================================================================================
		//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub_cumh_rows == 0){
				row = common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + common.in2_pad_cumv_sel2_collow - 1;
			d_in2_sub_cumh[p][ei_new] = d_in2_pad_cumv[p][ori_col*common.in2_pad_cumv_rows+ori_row];

		}

		}

		//==================================================
		//	SUBTRACTION
		//==================================================
		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){
		
		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_elem; ei_new++){

			// subtract
			d_in2_sub_cumh[p][ei_new] = d_in2_pad_cumv_sel[p][ei_new] - d_in2_sub_cumh[p][ei_new];

		}

		}

		//==================================================
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_rows; ei_new++){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+common.in2_sub_cumh_elem; position = position + common.in2_sub_cumh_rows){
				d_in2_sub_cumh[p][position] = d_in2_sub_cumh[p][position] + sum;
				sum = d_in2_sub_cumh[p][position];
			}

		}

		}

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_sel_elem; ei_new++)

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub_cumh_sel_rows == 0){
				row = common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + common.in2_sub_cumh_sel_collow - 1;
			d_in2_sub_cumh_sel[p][ei_new] = d_in2_sub_cumh[p][ori_col*common.in2_sub_cumh_rows+ori_row];

		}

		}

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub2_rows == 0){
				row = common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + common.in2_sub_cumh_sel2_collow - 1;
			d_in2_sub2[p][ei_new] = d_in2_sub_cumh[p][ori_col*common.in2_sub_cumh_rows+ori_row];

		}

		}

		//==================================================
		//	SUBTRACTION
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++)

			// subtract
			d_in2_sub2[p][ei_new] = d_in2_sub_cumh_sel[p][ei_new] - d_in2_sub2[p][ei_new];

		}

		}

		//======================================================================================================================================================
		//	CUMULATIVE SUM 2
		//======================================================================================================================================================

		//====================================================================================================
		//	MULTIPLICATION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sqr_elem; ei_new++)

			temp = d_in2[p][ei_new];
			d_in2_sqr[p][ei_new] = temp * temp;

		}

		}

		//====================================================================================================
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	PAD ARRAY
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_pad_cumv_elem; ei_new++){

			// figure out row/col location in padded array
			row = (ei_new+1) % common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_pad_cumv_rows == 0){
				row = common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (common.in2_pad_add_rows+common.in2_sqr_rows) && 
				col > (common.in2_pad_add_cols-1) && 
				col < (common.in2_pad_add_cols+common.in2_sqr_cols)){
				ori_row = row - common.in2_pad_add_rows;
				ori_col = col - common.in2_pad_add_cols;
				d_in2_pad_cumv[p][ei_new] = d_in2_sqr[p][ori_col*common.in2_sqr_rows+ori_row];
			}
			else{																							// do if otherwise
				d_in2_pad_cumv[p][ei_new] = 0;
			}

		}

		}

		//==================================================
		//	VERTICAL CUMULATIVE SUM
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		//work
		for(ei_new=0; ei_new<common.in2_pad_cumv_cols; ei_new++){

			// figure out column position
			pos_ori = ei_new*common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+common.in2_pad_cumv_rows; position = position + 1){
				d_in2_pad_cumv[p][position] = d_in2_pad_cumv[p][position] + sum;
				sum = d_in2_pad_cumv[p][position];
			}

		}

		}

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_pad_cumv_sel_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_pad_cumv_sel_rows == 0){
				row = common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + common.in2_pad_cumv_sel_collow - 1;
			d_in2_pad_cumv_sel[p][ei_new] = d_in2_pad_cumv[p][ori_col*common.in2_pad_cumv_rows+ori_row];

		}

		}

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub_cumh_rows == 0){
				row = common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + common.in2_pad_cumv_sel2_collow - 1;
			d_in2_sub_cumh[p][ei_new] = d_in2_pad_cumv[p][ori_col*common.in2_pad_cumv_rows+ori_row];

		}

		}

		//==================================================
		//	SUBTRACTION
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_elem; ei_new++){

			// subtract
			d_in2_sub_cumh[p][ei_new] = d_in2_pad_cumv_sel[p][ei_new] - d_in2_sub_cumh[p][ei_new];

		}

		}

		//==================================================
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_rows; ei_new++){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+common.in2_sub_cumh_elem; position = position + common.in2_sub_cumh_rows){
				d_in2_sub_cumh[p][position] = d_in2_sub_cumh[p][position] + sum;
				sum = d_in2_sub_cumh[p][position];
			}

		}

		}

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub_cumh_sel_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub_cumh_sel_rows == 0){
				row = common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + common.in2_sub_cumh_sel_collow - 1;
			d_in2_sub_cumh_sel[p][ei_new] = d_in2_sub_cumh[p][ori_col*common.in2_sub_cumh_rows+ori_row];

		}

		}

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in2_sub2_rows == 0){
				row = common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + common.in2_sub_cumh_sel2_collow - 1;
			d_in2_sqr_sub2[p][ei_new] = d_in2_sub_cumh[p][ori_col*common.in2_sub_cumh_rows+ori_row];

		}

		}

		//==================================================
		//	SUBTRACTION
		//==================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			// subtract
			d_in2_sqr_sub2[ei_new] = d_in2_sub_cumh_sel[ei_new] - d_in2_sqr_sub2[ei_new];

		}

		}

		//======================================================================================================================================================
		//	FINAL
		//======================================================================================================================================================

		//====================================================================================================
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			temp = d_in2_sub2[p][ei_new];
			temp2 = d_in2_sqr_sub2[p][ei_new] - (temp * temp / common.in_elem);
			if(temp2 < 0){
				temp2 = 0;
			}
			d_in2_sqr_sub2[p][ei_new] = sqrt(temp2);

		}

		}

		//====================================================================================================
		//	MULTIPLICATION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in_sqr_elem; ei_new++){

			temp = d_in[p][ei_new];
			d_in_sqr[p][ei_new] = temp * temp;

		}

		}

		//====================================================================================================
		//	IN SUM
		//====================================================================================================

		#pragma acc kernels loop reduction(+:in_final_sum)
		for (p = 0; p < ALL_POINTS; p++){

		// work
		#pragma acc loop
		for(ei_new=0; ei_new<common.in_cols; ei_new++){

			sum = 0;
			for(i = 0; i < common.in_rows; i++){

				sum = sum + d_in[p][ei_new*common.in_rows+i];

			}
			in_final_sum += sum;

		}

		}

		//====================================================================================================
		//	IN_SQR SUM
		//====================================================================================================

		#pragma acc kernels loop redduction(+:in_sqr_final_sum)
		for (p = 0; p < ALL_POINTS; p++){

		#pragma acc loop seq
		for(ei_new=0; ei_new<common.in_sqr_rows; ei_new++){
				
			sum = 0;
			for(i = 0; i < common.in_sqr_cols; i++){

				sum = sum + d_in_sqr[p][ei_new+common.in_sqr_rows*i];

			}
			in_sqr_final_sum += sum;

		}

		}

		//====================================================================================================
		//	DENOMINATOR T
		//====================================================================================================

		if(tx == 0){

			mean = in_final_sum / common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			variance  = (in_sqr_final_sum / common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			denomT = sqrt(float(common.in_elem-1))*deviation;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			d_in2_sqr_sub2[p][ei_new] = d_in2_sqr_sub2[p][ei_new] * denomT;

		}

		}

		//====================================================================================================
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.conv_elem; ei_new+=){

			d_conv[p][ei_new] = d_conv[p][ei_new] - d_in2_sub2[p][ei_new] * in_final_sum / common.in_elem;

		}

		}

		//====================================================================================================
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.in2_sub2_elem; ei_new++){

			d_in2_sqr_sub2[p][ei_new] = d_conv[p][ei_new] / d_in2_sqr_sub2[p][ei_new];

		}

		}

		//======================================================================================================================================================
		//	TEMPLATE MASK CREATE
		//======================================================================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		cent = common.sSize + common.tSize + 1;
		if(common_change.frame_no == 0){
			tMask_row = cent + d_Row[point_no] - d_Row[point_no] - 1;
			tMask_col = cent + d_Col[point_no] - d_Col[point_no] - 1;
		}
		else{
			pointer = common_change.frame_no-1+point_no*common.no_frames;
			tMask_row = cent + d_tRowLoc[pointer] - d_Row[point_no] - 1;
			tMask_col = cent + d_tColLoc[pointer] - d_Col[point_no] - 1;
		}


		//work
		for(ei_new=0; ei_new<common.tMask_elem; ei_new++){

			location = tMask_col*common.tMask_rows + tMask_row;

			if(ei_new==location){
				d_tMask[ei_new] = 1;
			}
			else{
				d_tMask[ei_new] = 0;
			}

		}

		}

		//======================================================================================================================================================
		//	MASK CONVOLUTION
		//======================================================================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		// work
		for(ei_new=0; ei_new<common.mask_conv_elem; ei_new++){

			// figure out row/col location in array
			ic = (ei_new+1) % common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % common.mask_conv_rows == 0){
				ic = common.mask_conv_rows;
				jc = jc-1;
			}

			//
			j = jc + common.mask_conv_joffset;
			jp1 = j + 1;
			if(common.mask_cols < jp1){
				ja1 = jp1 - common.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(common.tMask_cols < j){
				ja2 = common.tMask_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + common.mask_conv_ioffset;
			ip1 = i + 1;
			
			if(common.mask_rows < ip1){
				ia1 = ip1 - common.mask_rows;
			}
			else{
				ia1 = 1;
			}
			if(common.tMask_rows < i){
				ia2 = common.tMask_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_tMask[common.tMask_rows*(ja-1)+ia-1] * 1;
				}
			}

			// //d_mask_conv[common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_mask_conv[p][ei_new] = d_in2_sqr_sub2[p][ei_new] * s;

		}

		}

		//======================================================================================================================================================
		//	MAXIMUM VALUE
		//======================================================================================================================================================

		//====================================================================================================
		//	INITIAL SEARCH
		//====================================================================================================

		#pragma acc kernels
		for (p = 0; p < ALL_POINTS; p++){

		for(ei_new=0; ei_new<common.mask_conv_rows; ei_new++){

			for(i=0; i<common.mask_conv_cols; i++){
				largest_coordinate_current = ei_new*common.mask_conv_rows+i;
				largest_value_current = abs(d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
			}
			par_max_coo[ei_new] = largest_coordinate;
			par_max_val[ei_new] = largest_value;

		}

		}

		//====================================================================================================
		//	FINAL SEARCH
		//====================================================================================================

		if(tx == 0){

			for(i = 0; i < common.mask_conv_rows; i++){
				if(par_max_val[i] > fin_max_val){
					fin_max_val = par_max_val[i];
					fin_max_coo = par_max_coo[i];
				}
			}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % common.mask_conv_rows == 0){
				largest_row = common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - common.in_rows - (common.sSize - common.tSize);
			offset_col = largest_col - common.in_cols - (common.sSize - common.tSize);
			pointer = common_change.frame_no+point_no*common.no_frames;
			d_tRowLoc[pointer] = d_Row[point_no] + offset_row;
			d_tColLoc[pointer] = d_Col[point_no] + offset_col;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

	}
	
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	COORDINATE AND TEMPLATE UPDATE
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// time19 = clock();

	// if the last frame in the bath, update template
	if(common_change.frame_no != 0 && (common_change.frame_no)%10 == 0){

		// update coordinate
		loc_pointer = point_no*common.no_frames+common_change.frame_no;
		d_Row[point_no] = d_tRowLoc[loc_pointer];
		d_Col[point_no] = d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		while(ei_new < common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % common.in_rows == 0){
				row = common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_Row[point_no] - 25 + row - 1;
			ori_col = d_Col[point_no] - 25 + col - 1;
			ori_pointer = ori_col*common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = common.alpha*d_in[ei_new] + (1.00-common.alpha)*common_change.frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

	}

}

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	END OF FUNCTION
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
