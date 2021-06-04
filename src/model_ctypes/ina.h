#ifndef _INA_H_
#define _INA_H_

#include "math.h"
#include <stdio.h>

#define S_SIZE 6
#define C_SIZE 30
#define A_SIZE 11

void initialize_states_default(double *STATES);
void initialize_constants_default(double *CONSTANTS, double *args) ;
void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC);
void compute_rates(const double time,  double *STATES,  double *CONSTANTS,  double *ALGEBRAIC, double *RATES);

#endif
