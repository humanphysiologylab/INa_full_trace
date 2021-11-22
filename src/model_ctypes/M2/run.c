#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>  

#include "./ina.h"

#define Ith(v,i)    NV_Ith_S(v,i)
#define IJth(A,i,j) SM_ELEMENT_D(A, i, j)
#define T0    RCONST(0.0) 
#define RTOL  RCONST(1.0e-9)
#define ATOL  RCONST(1.0e-9)
// static void PrintFinalStats(void *cvode_mem);
// static int check_retval(void *returnvalue, const char *funcname, int opt);

int rhs(realtype t, N_Vector y, N_Vector ydot, void *data) {
        double *C = (double *)data, *A = ((double *)data) + C_SIZE;
        N_Vector V_C = N_VMake_Serial(C_SIZE, C);
        N_Vector V_A = N_VMake_Serial(A_SIZE, A);
        compute_rates(t, y, V_C, V_A, ydot);
        N_VDestroy_Serial(V_A);
        N_VDestroy_Serial(V_C);
        return 0;
}

int run(double *S, double *C,
        double *time_array, double *voltage_command_array, int array_length,
        double *output_S, double *output_A) {

        double data[C_SIZE + A_SIZE];
        double *A = data + C_SIZE;
        realtype t, tout, reltol;
        SUNMatrix Matrix;
        SUNLinearSolver LS;
        void *cvode_mem;
        int retval;

        for (int i = 0; i < C_SIZE; ++i) {
                data[i] = C[i];
        }

        N_Vector V_S = N_VMake_Serial(S_SIZE, S);
        N_Vector V_C = N_VMake_Serial(C_SIZE, C);
        N_Vector V_A = N_VMake_Serial(A_SIZE, A);
        
        compute_algebraic(T0, V_S, V_C, V_A);
        memcpy(output_S, NV_DATA_S(V_S), S_SIZE * sizeof(double));
        memcpy(output_A, NV_DATA_S(V_A), A_SIZE * sizeof(double));
        
        // INIT
        cvode_mem = CVodeCreate(CV_BDF);
        retval = CVodeInit(cvode_mem, rhs, T0, V_S);
        retval = CVodeSetUserData(cvode_mem, data);
        
        // TOLERANCES
        N_Vector abstol = N_VNew_Serial(S_SIZE);
        Ith(abstol,0) = ATOL;//v_comp
        Ith(abstol,1) = ATOL;//v_p
        Ith(abstol,2) = ATOL;//v_m
        Ith(abstol,3) = ATOL;//m
        Ith(abstol,4) = ATOL;//h
        Ith(abstol,5) = ATOL;//j
        Ith(abstol,6) = ATOL;
        reltol = RTOL;
        retval = CVodeSVtolerances(cvode_mem, reltol, abstol);
        retval = CVodeSetErrFile(cvode_mem, NULL);
        
        // Linear solver
        Matrix = SUNDenseMatrix(S_SIZE, S_SIZE);
        LS = SUNLinSol_Dense(V_S, Matrix);
        retval = CVodeSetLinearSolver(cvode_mem, LS, Matrix);
        // retval = CVodeSetErrFile(NULL, NULL);
        // t_stop = 
        retval = CVodeSetUserData(cvode_mem, data);
        for (int i = 1; i < array_length; i++) {
                tout = time_array[i];
                data[27] = voltage_command_array[i];
                retval = CVodeSetStopTime(cvode_mem, tout);
                
                // printf("At t = %0.4e    tout = %0.4e   data =%14.6e\n", t, tout, data[29]);
                retval = CVode(cvode_mem, tout, V_S, &t, CV_NORMAL);
                memcpy(output_S + i * S_SIZE, NV_DATA_S(V_S), S_SIZE * sizeof(double));
                memcpy(output_A + i * A_SIZE, NV_DATA_S(V_A), A_SIZE * sizeof(double));
        }
        // PrintFinalStats(cvode_mem);
        N_VDestroy_Serial(V_S);
        N_VDestroy_Serial(V_A);
        N_VDestroy_Serial(V_C);
        N_VDestroy_Serial(abstol);
        CVodeFree(&cvode_mem);
        SUNLinSolFree(LS);
        SUNMatDestroy(Matrix);

        return 2;
}

// static void PrintFinalStats(void *cvode_mem)
// {
//   long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
//   int retval;

//   retval = CVodeGetNumSteps(cvode_mem, &nst);
//   check_retval(&retval, "CVodeGetNumSteps", 1);
//   retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
//   check_retval(&retval, "CVodeGetNumRhsEvals", 1);
//   retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
//   check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
//   retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
//   check_retval(&retval, "CVodeGetNumErrTestFails", 1);
//   retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
//   check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
//   retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
//   check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

//   retval = CVodeGetNumJacEvals(cvode_mem, &nje);
//   check_retval(&retval, "CVodeGetNumJacEvals", 1);
//   retval = CVodeGetNumLinRhsEvals(cvode_mem, &nfeLS);
//   check_retval(&retval, "CVodeGetNumLinRhsEvals", 1);

//   retval = CVodeGetNumGEvals(cvode_mem, &nge);
//   check_retval(&retval, "CVodeGetNumGEvals", 1);

//   printf("\nFinal Statistics:\n");
//   printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
// 	 nst, nfe, nsetups, nfeLS, nje);
//   printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
// 	 nni, ncfn, netf, nge);
// }

// static int check_retval(void *returnvalue, const char *funcname, int opt)
// {
//   int *retval;

//   /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
//   if (opt == 0 && returnvalue == NULL) {
//     fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
// 	    funcname);
//     return(1); }

//   /* Check if retval < 0 */
//   else if (opt == 1) {
//     retval = (int *) returnvalue;
//     if (*retval < 0) {
//       fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
// 	      funcname, *retval);
//       return(1); }}

//   /* Check if function returned NULL pointer - no memory allocated */
//   else if (opt == 2 && returnvalue == NULL) {
//     fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
// 	    funcname);
//     return(1); }

//   return(0);
// }
