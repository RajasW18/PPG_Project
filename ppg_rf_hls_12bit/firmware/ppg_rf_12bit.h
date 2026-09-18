#ifndef PPG_RF_12BIT_H_
#define PPG_RF_12BIT_H_

#include "BDT.h"
#include "parameters.h"


// Prototype of top level function for C-synthesis
void ppg_rf_12bit(
	input_arr_t data,
	score_arr_t score,
	score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]);
#endif
