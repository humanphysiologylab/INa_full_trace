#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

        double t_out        = 0;
        double t            = 0;
        int ctx_state       = 0;

        memcpy(output_S, S, S_SIZE * sizeof(double));

        compute_algebraic(t, S, C, A);
        memcpy(output_A, A, A_SIZE * sizeof(double));


        double dt = 5e-8;

        for (int i = 1; i < array_length; i++) {
                t_out = time_array[i];
                data[31] = voltage_command_array[i];
                //lsoda(&ctx, S, &t, t_out);
                euler(&t, S, data, dt, t_out);
                memcpy(output_S + i * S_SIZE, S, S_SIZE * sizeof(double));
                memcpy(output_A + i * A_SIZE, A, A_SIZE * sizeof(double));

        }
        ctx_state = 2;


        return ctx_state;
}
