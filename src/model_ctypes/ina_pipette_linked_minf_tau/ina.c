#include "ina.h"
#include "math.h"

void initialize_states_default(double *STATES) {
  STATES[0] = -80;//v_comp
  STATES[1] = -80;//v_p
  STATES[2] = -80;//v_m
  STATES[3] = 0.;//m
  STATES[4] = 1.;//h
  STATES[5] = 1.;//j
  STATES[6] = 0;//I_out
}

void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC){//, double *RATES){

  ALGEBRAIC[0] =  1 / ((1 + exp(-(CONSTANTS[14] + STATES[2])/CONSTANTS[16]))*CONSTANTS[2]*exp( STATES[2] /CONSTANTS[3] ));
  //tau_m =  1 / ((1 + exp(-(v_half_m + v_m) / k_m))*a0_m * exp(v_m / s_m))
  ALGEBRAIC[1] = 1 / ((1 + exp((CONSTANTS[15] + STATES[2])/CONSTANTS[17]))*CONSTANTS[4]*exp(-STATES[2]/CONSTANTS[5]));
  //tau_h = 1 / ((1 + exp((v_half_h + v_m) / k_h)) * a0_h * exp(-v_m / s_h))
  ALGEBRAIC[2] = CONSTANTS[8] + 1 / ((1 + exp((CONSTANTS[15] + STATES[2])/CONSTANTS[17]))*CONSTANTS[6]*exp(-STATES[2]/CONSTANTS[7]));
  //tau_j = tau_j_const + 1 / ((1 + exp((v_half_h + v_m) / k_h))*a0_j * exp(-v_m / s_j))

  ALGEBRAIC[3] = 1 / (1 + exp(-(CONSTANTS[14] + STATES[2])/CONSTANTS[16]));
  //m_inf = 1 / (1 + exp(-(v_half_m + v_m) / k_m));
  ALGEBRAIC[4] = 1 / (1 + exp((CONSTANTS[15] + STATES[2])/CONSTANTS[17]));
  //h_inf = 1 / (1 + exp((v_half_h + v_m) / k_h));
  ALGEBRAIC[5] = CONSTANTS[23] + (CONSTANTS[23] - STATES[0])*(1/(1 - CONSTANTS[20]) - 1);
  //v_cp =  v_c + (v_c - v_comp)*(1/(1-alpha) - 1);
  ALGEBRAIC[6] = CONSTANTS[12] * STATES[2];
  //I_leak = g_leak * v_m;
  ALGEBRAIC[7] = CONSTANTS[11] * STATES[4] * pow(STATES[3],3) * STATES[5]* (STATES[2] - CONSTANTS[22]);
  //I_Na = g_max * h * pow(m,3) * (v_m - v_rev) * j ;

  ALGEBRAIC[8] = 1e9 * CONSTANTS[1] * ((STATES[1] + CONSTANTS[21] - STATES[2])/(CONSTANTS[9]*CONSTANTS[1]) - (1e-9)*(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1]);// RATES[2];
  //I_c = 1e9 * constants['c_m'] * (np.diff(df.v_m) / dt)
  //ALGEBRAIC[8] = 1e9 * CONSTANTS[1] * ((ALGEBRAIC[5] + CONSTANTS[21] - STATES[2])/(CONSTANTS[9]*CONSTANTS[1]) - (1e-9)*(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1]);// RATES[2];


  ALGEBRAIC[9] = 1e9 * CONSTANTS[0] *(ALGEBRAIC[5] - STATES[1])/(CONSTANTS[0]*CONSTANTS[10]);// RATES[1];
  //I_p = 1e9 * constants['c_p'] * (np.diff(df.v_p) / dt)
  ALGEBRAIC[10] = 1e9 * CONSTANTS[18] * CONSTANTS[1] *(CONSTANTS[23] - STATES[0])/(CONSTANTS[18]*CONSTANTS[1]*CONSTANTS[19]*CONSTANTS[9]*(1 - CONSTANTS[20]));// RATES[0]
  //I_comp = 1e9 * constants['x_c_comp']* constants['c_m'] * (np.diff(df.v_comp) / dt)


  ALGEBRAIC[11] = ALGEBRAIC[6] + ALGEBRAIC[7] + ALGEBRAIC[8]  + ALGEBRAIC[9]- ALGEBRAIC[10];
  //ALGEBRAIC[11] = ALGEBRAIC[6] + ALGEBRAIC[7] + ALGEBRAIC[8] - ALGEBRAIC[10];
}

void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES){

  compute_algebraic(time, STATES, CONSTANTS, ALGEBRAIC);

  RATES[0] = (CONSTANTS[23] - STATES[0])/(CONSTANTS[18]*CONSTANTS[1]*CONSTANTS[19]*CONSTANTS[9]*(1 - CONSTANTS[20]));
  //v_comp[i] = v_c[i-1] + (v_comp[i-1] - v_c[i-1]) * exp(-dt / tau_srp);
  //tau_srp = x_r_comp*r_m; * x_c_comp*c_m; * (1 - alpha);
  RATES[1] = (ALGEBRAIC[5] - STATES[1])/(CONSTANTS[0]*CONSTANTS[10]);
  //v_p[i] = v_cp[i-1] + (v_p[i-1] - v_cp[i-1]) * exp(-dt / (r_p * c_p));

  RATES[2] = (STATES[1] + CONSTANTS[21] - STATES[2])/(CONSTANTS[9]*CONSTANTS[1]) - (1e-9)*(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1];
  //v_m[i] = v_m[i-1] + (v_p[i-1] + v_off - v_m[i-1] ) * (dt / (r_m * c_m)) - 1e-9 * (I_Na[i-1] + I_leak[i-1]) * dt / c_m ;
  //it can be STATES[1] instead ALGEBRAIC[5]
  //RATES[2] = (ALGEBRAIC[5] + CONSTANTS[21] - STATES[2])/(CONSTANTS[9]*CONSTANTS[1]) - (1e-9)*(ALGEBRAIC[6] + ALGEBRAIC[7])/CONSTANTS[1];

  RATES[3] = (ALGEBRAIC[3] - STATES[3])/ALGEBRAIC[0];
  //m[i] = m_inf + (m[i-1] - m_inf) * exp(-dt/tau_m);
  RATES[4] = (ALGEBRAIC[4] - STATES[4])/ALGEBRAIC[1];
  //h[i] = h_inf + (h[i-1] - h_inf) * exp(-dt/tau_h);
  RATES[5] = (ALGEBRAIC[4] - STATES[5])/ALGEBRAIC[2];
  //j[i] = h_inf + (j[i-1] - h_inf) * exp(-dt/tau_j);
  RATES[6] = (ALGEBRAIC[11] - STATES[6])/CONSTANTS[13];
  //I_out = (I_in - I_out)/tau_z
  }
