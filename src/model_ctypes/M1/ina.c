#include <math.h>
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */
#include <stdio.h>
#include "ina.h"

#define Ith(v,i)    NV_Ith_S(v,i)
#define NM          RCONST(1.0e-9)
#define RNM          RCONST(1.0e9)

void initialize_states_default(N_Vector STATES){
  Ith(STATES,0) = -80;// v_comp
  Ith(STATES,1) = -80;// v_p
  Ith(STATES,2) = -80;// v_m
  Ith(STATES,3) = 0.;// m
  Ith(STATES,4) = 1.;// h
  Ith(STATES,5) = 1.;// j
  Ith(STATES,6) = 0;// I_out
}


void compute_algebraic(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC){
  realtype v_comp = Ith(STATES,0);
  realtype v_p = Ith(STATES,1);
  realtype v_m = Ith(STATES,2);
  realtype m = Ith(STATES,3);
  realtype h = Ith(STATES,4);
  realtype j = Ith(STATES,5);

  realtype tau_m = Ith(ALGEBRAIC,0);
  realtype tau_h = Ith(ALGEBRAIC,1);
  realtype tau_j = Ith(ALGEBRAIC,2);
  realtype m_inf = Ith(ALGEBRAIC,3);
  realtype h_inf = Ith(ALGEBRAIC,4);
  realtype v_cp = Ith(ALGEBRAIC,5);
  realtype I_leak = Ith(ALGEBRAIC,6);
  realtype I_Na = Ith(ALGEBRAIC,7);
  realtype I_c = Ith(ALGEBRAIC,8);
  realtype I_p = Ith(ALGEBRAIC,9);
  realtype I_comp = Ith(ALGEBRAIC,10);
  realtype I_in = Ith(ALGEBRAIC,11);

  realtype c_p = Ith(CONSTANTS,0);
  realtype c_m = Ith(CONSTANTS,1);
  realtype a0_m = Ith(CONSTANTS,2);
  realtype b0_m = Ith(CONSTANTS,3);
  realtype delta_m = Ith(CONSTANTS,4);
  realtype s_m = Ith(CONSTANTS,5);
  realtype tau_m_const = Ith(CONSTANTS,6);
  realtype a0_h = Ith(CONSTANTS,7);
  realtype b0_h = Ith(CONSTANTS,8);
  realtype delta_h = Ith(CONSTANTS,9);
  realtype s_h = Ith(CONSTANTS,10);
  realtype tau_h_const = Ith(CONSTANTS,11);
  realtype a0_j = Ith(CONSTANTS,12);
  realtype b0_j = Ith(CONSTANTS,13);
  realtype delta_j = Ith(CONSTANTS,14);
  realtype s_j = Ith(CONSTANTS,15);
  realtype tau_j_const = Ith(CONSTANTS,16);
  realtype R = Ith(CONSTANTS,17);
  realtype R_f = Ith(CONSTANTS,18);
  realtype g_max = Ith(CONSTANTS,19);
  realtype g_leak = Ith(CONSTANTS,20);
  realtype v_half_m = Ith(CONSTANTS,22);
  realtype v_half_h = Ith(CONSTANTS,23);
  realtype k_m = Ith(CONSTANTS,24);
  realtype k_h = Ith(CONSTANTS,25);
  realtype x_c_comp = Ith(CONSTANTS,26);
  realtype x_r_comp = Ith(CONSTANTS,27);
  realtype alpha = Ith(CONSTANTS,28);
  realtype v_off = Ith(CONSTANTS,29);
  realtype v_rev = Ith(CONSTANTS,30);
  realtype v_c = Ith(CONSTANTS,31);

  tau_m = tau_m_const + 1/(a0_m * exp(v_m / s_m) + b0_m * exp(- v_m / delta_m));
  tau_h = tau_h_const + 1/(a0_h * exp(- v_m / s_h) + b0_h * exp(v_m / delta_h));
  tau_j = tau_j_const + 1/(a0_j * exp(- v_m / s_j) + b0_j * exp(v_m / delta_j));
  
  m_inf = 1 / (1 + exp(-(v_half_m + v_m) / k_m));
  h_inf = 1 / (1 + exp((v_half_h + v_m) / k_h));
  
  v_cp = v_c + (v_c - v_comp) * (1 / (1 - alpha) - 1);  
  
  I_leak = g_leak * v_m;
  I_Na = g_max * h * pow(m,3) * j * (v_m - v_rev);
  I_c = RNM * c_m * ((v_p + v_off - v_m) / (R * c_m) - NM * (I_leak + I_Na) / c_m);// 1e9 * c_m * dv_m / dt
  I_p = RNM * c_p *(v_cp - v_p) / (c_p * R_f);// 1e9 * c_p * dv_p / dt
  I_comp = RNM * x_c_comp * c_m * (v_c - v_comp)/(x_c_comp * c_m * x_r_comp * R * (1 - alpha));// 1e9 * x_c_comp * c_m * d v_comp / dt
  I_in = I_leak + I_Na + I_c + I_p - I_comp;

  Ith(ALGEBRAIC,0) = tau_m;
  Ith(ALGEBRAIC,1) = tau_h;
  Ith(ALGEBRAIC,2) = tau_j;
  Ith(ALGEBRAIC,3) = m_inf;
  Ith(ALGEBRAIC,4) = h_inf;
  Ith(ALGEBRAIC,5) = v_cp;
  Ith(ALGEBRAIC,6) = I_leak;
  Ith(ALGEBRAIC,7) = I_Na;
  Ith(ALGEBRAIC,8) = I_c;
  Ith(ALGEBRAIC,9) = I_p;
  Ith(ALGEBRAIC,10) = I_comp;
  Ith(ALGEBRAIC,11) = I_in;
}

void compute_rates(const realtype time,  N_Vector STATES, N_Vector CONSTANTS,  N_Vector ALGEBRAIC, N_Vector RATES){

  compute_algebraic(time, STATES, CONSTANTS, ALGEBRAIC);

  realtype v_comp = Ith(STATES,0);
  realtype v_p = Ith(STATES,1);
  realtype v_m = Ith(STATES,2);
  realtype m = Ith(STATES,3);
  realtype h = Ith(STATES,4);
  realtype j = Ith(STATES,5);
  realtype I_out = Ith(STATES,6);

  realtype tau_m = Ith(ALGEBRAIC,0);
  realtype tau_h = Ith(ALGEBRAIC,1);
  realtype tau_j = Ith(ALGEBRAIC,2);
  realtype m_inf = Ith(ALGEBRAIC,3);
  realtype h_inf = Ith(ALGEBRAIC,4);
  realtype v_cp = Ith(ALGEBRAIC,5);
  realtype I_leak = Ith(ALGEBRAIC,6);
  realtype I_Na = Ith(ALGEBRAIC,7);
  realtype I_in = Ith(ALGEBRAIC,11);

  realtype c_p = Ith(CONSTANTS,0);
  realtype c_m = Ith(CONSTANTS,1);
  realtype R = Ith(CONSTANTS,17);
  realtype R_f = Ith(CONSTANTS,18);
  realtype tau_z = Ith(CONSTANTS,21);
  realtype x_c_comp = Ith(CONSTANTS,26);
  realtype x_r_comp = Ith(CONSTANTS,27);
  realtype alpha = Ith(CONSTANTS,28);
  realtype v_off = Ith(CONSTANTS,29);
  realtype v_c = Ith(CONSTANTS,31);

  
  Ith(RATES,0) = (v_c - v_comp) / (x_c_comp * c_m * x_r_comp * R * (1 - alpha));// v_comp
  Ith(RATES,1) = (v_cp - v_p) / (c_p * R_f); // v_p
  Ith(RATES,2) = (v_p + v_off - v_m) / (R * c_m) - NM * (I_leak + I_Na) / c_m;// v_m
  Ith(RATES,3) = (m_inf - m)/tau_m;// m
  Ith(RATES,4) = (h_inf - h)/tau_h;// h
  Ith(RATES,5) = (h_inf - j)/tau_j;// j
  Ith(RATES,6) = (I_in - I_out)/tau_z;// I_out
  }

