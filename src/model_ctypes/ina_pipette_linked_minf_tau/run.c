#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../liblsoda/src/common.h"
#include "../../liblsoda/src/lsoda.h"
#include "./ina.h"

int rhs(double t, double *y, double *ydot, void *data) {

        double *C = (double *)data, *A = ((double *)data) + C_SIZE;
        compute_rates(t, y, C, A, ydot);
        return 0;
}


int euler(double *t, double *y, void *data,
          double dt, double t_out) {

        double ydot[S_SIZE];

        while (*t < t_out) {

                rhs(*t, y, ydot, data);

                if (*t + dt <= t_out) {
                        *t += dt;
                } else {
                        dt = t_out - *t;
                        *t = t_out;
                }

                for (int i = 0; i < S_SIZE; ++i) {
                        y[i] += dt * ydot[i];
                }
        }

        return 0;
}


int run(double *S, double *C,
        double *time_array, double *voltage_command_array, int array_length,
        double *output_S, double *output_A) {

        double data[C_SIZE + A_SIZE];
        double *A = data + C_SIZE;

        for (int i = 0; i < C_SIZE; ++i) {
                data[i] = C[i];
        }

        double atol[S_SIZE], rtol[S_SIZE];
        int neq = S_SIZE;

        struct lsoda_opt_t opt = {0};
        opt.ixpr    = 0;
        opt.rtol    = rtol;
        opt.atol    = atol;
        opt.itask   = 1;
        //opt.hmin    = 1e-12;
        opt.hmax    = 5e-5;

        double atol_mult[] = { /*v_comp*/ 1e-2, /*v_p*/ 1e-2, /*v_m*/ 1e-2,
                                          /*m*/ 1e-4, /*h*/ 1e-4, /*j*/ 1e-4,
                                          /*I_out*/ 1e-2};

        for (int i = 0; i < S_SIZE; ++i) {
                rtol[i] = 5e-5;
                atol[i] = atol_mult[i];
        }

        double t               = 0;
        int ctx_state       = 0;

        memcpy(output_S, S, S_SIZE * sizeof(double));

        compute_algebraic(t, S, C, A);
        memcpy(output_A, A, A_SIZE * sizeof(double));

        struct lsoda_context_t ctx = {
                .function = rhs,
                .neq = neq,
                .data = data,
                .state = 1,
        };
        lsoda_prepare(&ctx, &opt);

        for (int i = 1; i < array_length; i++) {
                //t = time_array[i-1];
                double t_out = time_array[i];
                data[23] = voltage_command_array[i];
                lsoda(&ctx, S, &t, t_out);
                memcpy(output_S + i * S_SIZE, S, S_SIZE * sizeof(double));
                memcpy(output_A + i * A_SIZE, A, A_SIZE * sizeof(double));

                if (ctx.state != 2) {
                        return ctx.state;
                }

        }

        ctx_state = ctx.state;
        lsoda_free(&ctx);

        return ctx_state;
}
