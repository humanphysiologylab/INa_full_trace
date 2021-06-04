//#include <iostream>
#include "ina.h"
#include "math.h"

void initialize_states_default(double *STATES) {
  STATES[0] = -80;//v_comp
  STATES[1] = -80;//v_p
  STATES[2] = -80;//v_m
  STATES[3] = 0.;//m
  STATES[4] = 1.;//h
  STATES[5] = 1.;//j
}
void initialize_constants_default(double *CONSTANTS, double *args) {
  CONSTANTS[0] = args[0];//c_p
  CONSTANTS[1] = args[1];//c_m
  CONSTANTS[2] = args[2];//ao_m
  CONSTANTS[3] = args[3];//b0_m
  CONSTANTS[4] = args[4];//delta_m
  CONSTANTS[5] = args[5];//s_m
  CONSTANTS[6] = args[6];//a0_h
  CONSTANTS[7] = args[7];//b0_h
  CONSTANTS[8] = args[8];//delta_h
  CONSTANTS[9] = args[9];//s_h
  CONSTANTS[10] = args[10];//a0_j
  CONSTANTS[11] = args[11];//b0_j
  CONSTANTS[12] = args[12];//delta_j
  CONSTANTS[13] = args[13];//s_j
  CONSTANTS[14] = args[14];//tau_j_const
  CONSTANTS[15] = args[15];//r_m
  CONSTANTS[16] = args[16];//r_p
  CONSTANTS[17] = args[17];//g_max
  CONSTANTS[18] = args[18];//g_leak
  CONSTANTS[19] = args[19];//tau_z
  CONSTANTS[20] = args[20];//v_half_m
  CONSTANTS[21] = args[21];//v_half_h
  CONSTANTS[22] = args[22];//k_m
  CONSTANTS[23] = args[23];//k_h
  CONSTANTS[24] = args[24];//x_c_comp
  CONSTANTS[25] = args[25];//x_r_comp
  CONSTANTS[26] = args[26];//alpha
  CONSTANTS[27] = args[27];//v_off
  CONSTANTS[28] = args[28];//v_rev
  CONSTANTS[29] = args[29];//v_c;
  }
void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC){//, double *RATES){
  ALGEBRAIC[0] = 1/(CONSTANTS[2] * exp(STATES[2]/CONSTANTS[5]) + CONSTANTS[3]*exp(- STATES[2]/CONSTANTS[4]));
  //tau_m = 1 / (a0_m * exp(v_m[i-1] / (s_m)) + b0_m * exp(v_m[i-1] / (-delta_m)))
  ALGEBRAIC[1] = 1/(CONSTANTS[6] * exp(- STATES[2]/CONSTANTS[9]) + CONSTANTS[7]*exp(STATES[2]/CONSTANTS[8]));
  //tau_h = 1 / (a0_h * exp(v_m[i-1] / (-s_h)) + b0_h * exp(v_m[i-1] / (delta_h)))
  ALGEBRAIC[2] = CONSTANTS[14] + 1/(CONSTANTS[10] * exp( STATES[2]/CONSTANTS[13]) + CONSTANTS[11]*exp(-STATES[2]/CONSTANTS[12]));
  //tau_j = tau_j_const + 1 / (a0_j * exp(v_m[i-1] / (s_j)) + b0_j * exp(v_m[i-1] / (-delta_j)))
  ALGEBRAIC[3] = 1 / (1 + exp((-CONSTANTS[20] - STATES[2])/CONSTANTS[22]));
  //m_inf = 1 / (1 + exp((- v_half_m - v_m[i]) / k_m));
  ALGEBRAIC[4] = 1 / (1 + exp((CONSTANTS[21] + STATES[2])/CONSTANTS[23]));
  //h_inf = 1 / (1 + exp((v_half_h + v_m[i]) / k_h));
  ALGEBRAIC[5] = CONSTANTS[29] + (CONSTANTS[29] - STATES[0])*(1/(1 - CONSTANTS[26]) - 1);
  //v_cp[i] =  v_c[i] + (v_c[i] - v_comp[i])*(1/(1-alpha) - 1);
  ALGEBRAIC[6] = CONSTANTS[18] * STATES[2];
  //I_leak[i] = g_leak * v_m[i];
  ALGEBRAIC[7] = CONSTANTS[17] * STATES[4] * pow(STATES[3],3) * STATES[5]* (STATES[2] - CONSTANTS[28]);
  //I_Na[i] = g_max * h[i] * pow(m[i],3) * (v_m[i] - v_rev) * j[i] ;
  ALGEBRAIC[8] = 1e9 * CONSTANTS[1] * (ALGEBRAIC[5] + CONSTANTS[27] - STATES[2])/(CONSTANTS[15]*CONSTANTS[1]) - CONSTANTS[1] *(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1];// RATES[2];
  //I_c = 1e9 * constants['c_m'] * (np.diff(df.v_m) / dt)
  ALGEBRAIC[9] = 1e9 * CONSTANTS[0] *(ALGEBRAIC[5] - STATES[1])/(CONSTANTS[0]*CONSTANTS[16]);// RATES[1];
  //I_p = 1e9 * constants['c_p'] * (np.diff(df.v_p) / dt)
  ALGEBRAIC[10] = 1e9 * CONSTANTS[24] * CONSTANTS[1] *(CONSTANTS[29] - STATES[0])/(CONSTANTS[24]*CONSTANTS[1]*CONSTANTS[25]*CONSTANTS[15]*(1 - CONSTANTS[26]));// RATES[0]
  //I_comp = 1e9 * constants['x_c_comp']* constants['c_m'] * (np.diff(df.v_comp) / dt)
}

void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES){

  RATES[0] = (CONSTANTS[29] - STATES[0])/(CONSTANTS[24]*CONSTANTS[1]*CONSTANTS[25]*CONSTANTS[15]*(1 - CONSTANTS[26]));
  //v_comp[i] = v_c[i-1] + (v_comp[i-1] - v_c[i-1]) * exp(-dt / tau_srp);
  //tau_srp = x_r_comp*r_m; * x_c_comp*c_m; * (1 - alpha);
  RATES[1] = (ALGEBRAIC[5] - STATES[1])/(CONSTANTS[0]*CONSTANTS[16]);
  //v_p[i] = v_cp[i-1] + (v_p[i-1] - v_cp[i-1]) * exp(-dt / (r_p * c_p));
  //warning!  v_c instead v_cp we can do now, dont forget!!!
  RATES[2] = (ALGEBRAIC[5] + CONSTANTS[27] - STATES[2])/(CONSTANTS[15]*CONSTANTS[1]) - (1e-9)*(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1];
  //v_m[i] = v_m[i-1] + (v_cp[i-1] + v_off - v_m[i-1] ) * (dt / (r_m * c_m)) - 1e-9 * (I_Na[i-1] + I_leak[i-1]) * dt / c_m ;
  //(STATES[1]
  RATES[3] = (ALGEBRAIC[3] - STATES[3])/ALGEBRAIC[0];
  //m[i] = m_inf + (m[i-1] - m_inf) * exp(-dt/tau_m);
  RATES[4] = (ALGEBRAIC[4] - STATES[4])/ALGEBRAIC[1];
  //h[i] = h_inf + (h[i-1] - h_inf) * exp(-dt/tau_h);
  RATES[5] = (ALGEBRAIC[4] - STATES[5])/ALGEBRAIC[2];
  //j[i] = h_inf + (j[i-1] - h_inf) * exp(-dt/tau_j);
  }
