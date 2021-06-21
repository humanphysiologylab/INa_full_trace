#ifndef _INA_MIRAMS_H_
#define _INA_MIRAMS_H_

#define S_SIZE 7
#define C_SIZE 32
#define A_SIZE 11

void initialize_states_default(double *STATES);
void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC);
void compute_rates(const double time,  double *STATES,  double *CONSTANTS,  double *ALGEBRAIC, double *RATES);

#endif
