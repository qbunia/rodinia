//=====================================================================
//	MAIN FUNCTION
//=====================================================================
inline void fin(	int timeinst,
													fp* initvalu,
													fp* finavalu,
													int offset_ecc,
													int offset_Dyad,
													int offset_SL,
													int offset_Cyt,
													fp* params,
													fp* com){

//=====================================================================
//	VARIABLES
//=====================================================================

	// input parameters
	fp BtotDyad;
	fp CaMKIItotDyad;

	// compute variables
	fp Vmyo;																			// [L]
	fp Vdyad;																			// [L]
	fp VSL;																				// [L]
	// fp kDyadSL;																			// [L/msec]
	fp kSLmyo;																			// [L/msec]
	fp k0Boff;																			// [s^-1] 
	fp k0Bon;																			// [uM^-1 s^-1] kon = koff/Kd
	fp k2Boff;																			// [s^-1] 
	fp k2Bon;																			// [uM^-1 s^-1]
	// fp k4Boff;																			// [s^-1]
	fp k4Bon;																			// [uM^-1 s^-1]
	fp CaMtotDyad;
	fp Bdyad;																			// [uM dyad]
	fp J_cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca2cam_dyadSL;																	// [uM/msec dyad]
	fp J_ca4cam_dyadSL;																	// [uM/msec dyad]
	fp J_cam_SLmyo;																		// [umol/msec]
	fp J_ca2cam_SLmyo;																	// [umol/msec]
	fp J_ca4cam_SLmyo;																	// [umol/msec]

//=====================================================================
//	COMPUTATION
//=====================================================================

	// input parameters
	BtotDyad = params[1];
	CaMKIItotDyad = params[2];

	// ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
	finavalu[offset_ecc+35] = finavalu[offset_ecc+35] + 1e-3*com[0];
	finavalu[offset_ecc+36] = finavalu[offset_ecc+36] + 1e-3*com[1];
	finavalu[offset_ecc+37] = finavalu[offset_ecc+37] + 1e-3*com[2]; 

	// incorporate CaM diffusion between compartments
	Vmyo = 2.1454e-11;																// [L]
	Vdyad = 1.7790e-14;																// [L]
	VSL = 6.6013e-13;																// [L]
	// kDyadSL = 3.6363e-16;															// [L/msec]
	kSLmyo = 8.587e-15;																// [L/msec]
	k0Boff = 0.0014;																// [s^-1] 
	k0Bon = k0Boff/0.2;																// [uM^-1 s^-1] kon = koff/Kd
	k2Boff = k0Boff/100;															// [s^-1] 
	k2Bon = k0Bon;																	// [uM^-1 s^-1]
	// k4Boff = k2Boff;																// [s^-1]
	k4Bon = k0Bon;																	// [uM^-1 s^-1]
	CaMtotDyad = initvalu[offset_Dyad+0]
			   + initvalu[offset_Dyad+1]
			   + initvalu[offset_Dyad+2]
			   + initvalu[offset_Dyad+3]
			   + initvalu[offset_Dyad+4]
			   + initvalu[offset_Dyad+5]
			   + CaMKIItotDyad * (	  initvalu[offset_Dyad+6]
												  + initvalu[offset_Dyad+7]
												  + initvalu[offset_Dyad+8]
												  + initvalu[offset_Dyad+9])
			   + initvalu[offset_Dyad+12]
			   + initvalu[offset_Dyad+13]
			   + initvalu[offset_Dyad+14];
	Bdyad = BtotDyad - CaMtotDyad;																				// [uM dyad]
	J_cam_dyadSL = 1e-3 * (  k0Boff*initvalu[offset_Dyad+0] - k0Bon*Bdyad*initvalu[offset_SL+0]);			// [uM/msec dyad]
	J_ca2cam_dyadSL = 1e-3 * (  k2Boff*initvalu[offset_Dyad+1] - k2Bon*Bdyad*initvalu[offset_SL+1]);		// [uM/msec dyad]
	J_ca4cam_dyadSL = 1e-3 * (  k2Boff*initvalu[offset_Dyad+2] - k4Bon*Bdyad*initvalu[offset_SL+2]);		// [uM/msec dyad]

	J_cam_SLmyo = kSLmyo * (  initvalu[offset_SL+0] - initvalu[offset_Cyt+0]);								// [umol/msec]
	J_ca2cam_SLmyo = kSLmyo * (  initvalu[offset_SL+1] - initvalu[offset_Cyt+1]);							// [umol/msec]
	J_ca4cam_SLmyo = kSLmyo * (  initvalu[offset_SL+2] - initvalu[offset_Cyt+2]);							// [umol/msec]

	// ADJUST CAM Dyad 
	finavalu[offset_Dyad+0] = finavalu[offset_Dyad+0] - J_cam_dyadSL;
	finavalu[offset_Dyad+1] = finavalu[offset_Dyad+1] - J_ca2cam_dyadSL;
	finavalu[offset_Dyad+2] = finavalu[offset_Dyad+2] - J_ca4cam_dyadSL;

	// ADJUST CAM Sl
	finavalu[offset_SL+0] = finavalu[offset_SL+0] + J_cam_dyadSL*Vdyad/VSL - J_cam_SLmyo/VSL;
	finavalu[offset_SL+1] = finavalu[offset_SL+1] + J_ca2cam_dyadSL*Vdyad/VSL - J_ca2cam_SLmyo/VSL;
	finavalu[offset_SL+2] = finavalu[offset_SL+2] + J_ca4cam_dyadSL*Vdyad/VSL - J_ca4cam_SLmyo/VSL;

	// ADJUST CAM Cyt 
	finavalu[offset_Cyt+0] = finavalu[offset_Cyt+0] + J_cam_SLmyo/Vmyo;
	finavalu[offset_Cyt+1] = finavalu[offset_Cyt+1] + J_ca2cam_SLmyo/Vmyo;
	finavalu[offset_Cyt+2] = finavalu[offset_Cyt+2] + J_ca4cam_SLmyo/Vmyo;

}
