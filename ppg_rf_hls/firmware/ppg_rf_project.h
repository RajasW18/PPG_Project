#ifndef PPG_RF_PROJECT_H_
#define PPG_RF_PROJECT_H_

#include "BDT.h"
#include "parameters.h"

// Top level function for C-synthesis with individual feature inputs
void ppg_rf_infer(
    input_t x0, input_t x1, input_t x2, input_t x3, input_t x4,
    input_t x5, input_t x6, input_t x7, input_t x8, input_t x9,
    input_t x10, input_t x11, input_t x12, input_t x13, input_t x14,
    bool &prediction
);

#endif