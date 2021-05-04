import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd

from datetime import datetime
import future
from memory_profiler import profile
import line_profiler

from sklearn.metrics import mean_squared_error as MSE
from scipy import optimize as scop
from numba import njit



@profile
def rush_larsen_easy_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = y[i-1] + (x[i-1] - y[i-1]) * np.exp(c)
@profile
def euler_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = x[i-1] + (y[i-1] - x[i-1]) * c


@njit
def rush_larsen_easy_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = y[i-1] + (x[i-1] - y[i-1]) * np.exp(c)

@njit
def euler_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = x[i-1] + (y[i-1] - x[i-1]) * c

#@profile
#@njit

@profile
@njit
def rush_larsen_easy_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = y[i-1] + (x[i-1] - y[i-1]) * np.exp(c)

@njit
def euler_numba_helper(x, y, c):
    for i in range(1, len(x)):
        x[i] = x[i-1] + (y[i-1] - x[i-1]) * c

#@profile
#@njit

@profile
@njit
def calculate_circle(n, t, v_c, v_rev,v_cp,v_p,v_m,v_comp, m_inf,h_inf , m, h, j,I_leak, I_Na,args):

    c_p, c_m, a0_m, b0_m, delta_m, s_m, a0_h, b0_h, delta_h, s_h, a0_j, b0_j, delta_j, s_j,tau_j_const,\
    r_m, r_p, g_max, g_leak, v_half_m, v_half_h, k_m, k_h, x_c_comp, x_r_comp, alpha, v_off = args

    n_step = len(t)

    dt = t[1] - t[0]
    #I_leak[0] = g_leak * v_m[0]
    time_ = 25000

    n_start = 15

    I_leak[0] = g_leak * v_m[0]
    c_comp = x_c_comp*c_m
    r_comp = x_r_comp*r_m

    i=1
    circle = 0
    while i!= n_step:

        alpha_m  = a0_m * np.exp( v_m[i-1] / (s_m))
        beta_m = b0_m * np.exp( v_m[i-1] / (-delta_m))

        alpha_h  = a0_h * np.exp(v_m[i-1] / (-s_h))
        beta_h = b0_h * np.exp( v_m[i-1] / delta_h)

        #if False:
        #    alpha_m  = a0_m * np.exp( v_m / (s_m))
        #    beta_m = b0_m * np.exp( v_m / (-delta_m))

        #    alpha_h  = a0_h * np.exp(v_m / (s_h))
        #    beta_h = b0_h * np.exp( v_m/ (-delta_h))

        alpha_j  = a0_j * np.exp( v_m[i-1] / (s_j))
        beta_j = b0_j * np.exp( v_m[i-1] / (-delta_j))

        tau_m = 1 / (beta_m + alpha_m)#+0.000037
        tau_h = 1 / (beta_h + alpha_h)#+0.0002
        tau_j = tau_j_const + 1 / (beta_j + alpha_j)

        betta = 1/(1-alpha) - 1


        tau_srp = r_m * c_m * (1 - alpha)
        v_comp[i] = v_c[i-1] + (v_comp[i-1] - v_c[i-1]) * np.exp(-dt / tau_srp)
        v_cp[i] =  v_c[i-1] + (v_c[i-1] - v_comp[i-1])*betta
        v_p[i] = v_cp[i-1] + (v_p[i-1] - v_cp[i-1]) * np.exp(-dt / (r_p * c_p))
        v_m[i] = v_m[i-1] + (v_p[i-1] + v_off - v_m[i-1] ) * (dt / (r_m * c_m)) - 1e-9 * (I_Na[i-1] + I_leak[i-1]) * dt / c_m

        m[i] = m_inf + (m[i-1] - m_inf) * np.exp(-dt/tau_m)
        h[i] = h_inf + (h[i-1] - h_inf) * np.exp(-dt/tau_h)
        j[i] = h_inf + (j[i-1] - h_inf) * np.exp(-dt/tau_j)

        m_inf = 1 / (1 + np.exp((- v_half_m - v_m[i]) / k_m))
        h_inf = 1 / (1 + np.exp((v_half_h + v_m[i]) / k_h))

        I_leak[i] = g_leak * v_m[i]
        I_Na[i] = g_max * h[i] * (m[i]**3) * (v_m[i] - v_rev) * j[i]
        if (i-1)/time_ == (i-1)//time_ :
            while circle!= n_start:
                v_cp[i-1], v_p[i-1], v_m[i-1], v_comp[i-1], m_inf, h_inf, m[i-1], h[i-1], j[i-1], I_leak[i-1], I_Na[i-1] = \
                v_cp[i], v_p[i], v_m[i], v_comp[i], m_inf, h_inf, m[i], h[i], j[i], I_leak[i], I_Na[i]
                
                circle += 1
            i+=1
        else:
            i+=1
            circle = 0
    return v_cp,  v_p, v_m, v_comp, I_leak[::n], I_Na[::n]#tau_m[::n], tau_h[::n], tau_j[::n],


@profile
def calculate_I_out(x, *args):#, s0, c, protocol, ...):
    #print(x)
    y = x.copy()
    kwargs = args[-1]

    t = kwargs['t']
    dt = t[1] - t[0]

    n = int(5e-5/dt)

    v_list = kwargs['v_list']
    k_list = kwargs['k_list'] * n

    if kwargs.get('log_scale', False):
        y[:-1] = np.exp(y[:-1])
        #y = np.exp(y)
        assert np.all(y[:-1] > 0)
    #c_p, c_m, a0_m, b0_m, delta_m, s_m, a0_h, b0_h, delta_h, s_h, a0_j, b0_j, delta_j, s_j,tau_j_const,\
    #r_m, r_p, g_max, g_leak, v_half_m, v_half_h, k_m, k_h, x_c_comp, x_r_comp, alpha, v_off = y



    count = np.zeros_like(t)
    count[k_list] = 1
    count = np.cumsum(count)


    v_c = np.zeros_like(t)
    v_c = v_list[count.astype(int)]

    v_rev = 18


    #m_inf = np.zeros_like(t)
    #h_inf = np.zeros_like(t)
    #
    v_cp = np.zeros_like(t)
    v_p = np.zeros_like(t)
    v_m = np.zeros_like(t)
    v_comp = np.zeros_like(t)

    m = np.zeros_like(t)
    h = np.zeros_like(t)
    j = np.zeros_like(t)

    I_leak = np.zeros_like(t)
    I_Na = np.zeros_like(t)

    v_p[0] = -80
    v_m[0] = -80
    v_comp[0] =  - 80
    v_cp[0] =  - 80

    m_inf = 0
    h_inf = 1

    m[0] = 0
    h[0] = 1
    #n_step = len(t)

    #dt = t[1] - t[0]
    #time_ = 25000
    #n_start = 15000


    I_error = np.zeros(len(t)//n)
    try:
        v_cp, v_p, v_m, v_comp, I_leak, I_Na = calculate_circle(n, t, v_c, v_rev,v_cp,v_p,v_m,v_comp, m_inf,h_inf ,m, h, j,I_leak, I_Na,y)
    except ZeroDivisionError:
        I_error+=1e100
        return I_error
    c_p = y[0]
    c_m = y[1]
    x_c_comp = y[-4]

    I_c = 1e9 * c_m * (np.diff(v_m) / dt)[::n]
    I_p = 1e9 * c_p * (np.diff(v_p) / dt)[::n]
    I_comp=1e9 * x_c_comp* c_m * (np.diff(v_comp) / dt)[::n]

    if len(I_c) != len(I_Na):
        I_c = np.concatenate((I_c,I_c[-1:]))
        I_p = np.concatenate((I_p,I_p[-1:]))
        I_comp = np.concatenate((I_comp,I_comp[-1:]))

    tau_z = 5e-4 # 1e-12 * 5e8

    I_in = I_c  + I_leak + I_Na + I_p - I_comp
    I_out = np.zeros_like(I_in)

    I_out[0] = I_in[0]
    rush_larsen_easy_numba_helper(I_out, I_in, - dt * n / tau_z)
    #euler_numba_helper(I_out,I_in,(dt / tau_z))

    if kwargs.get('graph', True):
        #plt.plot(V_m_list, label = 'command')
        plt.figure()

        plt.plot(v_c, label = 'command')
        #plt.plot(v_comp, label = 'compensated')
        plt.plot(v_cp, label = 'prediction')
        plt.plot(v_p, label = 'pipette', ls = '--')
        plt.plot(v_m, label = 'membrane',ls = '-.')
        plt.legend()
        v_graph = np.arange(-95,35)


        plt.figure()
        plt.plot( m_inf, label = 'm_inf')
        plt.plot( h_inf, label = 'h_inf')
        plt.legend()

        #plt.figure()
        #tau_m_graph = 1 / (b0_m * np.exp((1-delta_m) * v_graph / (-s_m))
        #                  + a0_m * np.exp( v_m[i-1] / (s_m))
        #tau_h_graph = 1 / (b0_h * np.exp((1-delta_h) * v_graph / (-s_h))
        #                  + a0_h * np.exp(-delta_h * v_graph / (-s_h)))

        #tau_m_graph = 1 / (a0_m * np.exp( v_graph / (s_m))
         #                 +  b0_m * np.exp( v_graph / (-delta_m)))
        #tau_h_graph = 1 / (a0_h * np.exp(v_graph / (-s_h))
         #                 +  b0_h * np.exp( v_graph / (delta_h)))
        #tau_j_graph = tau_j_const + 1 / (a0_j * np.exp(v_graph / (s_j))
         #                 +  b0_j * np.exp( v_graph / (-delta_j)))
        #plt.plot(v_graph, tau_m_graph, label = 'tau_m')
        #plt.plot(v_graph, tau_h_graph, label = 'tau_h')
        #plt.plot(v_graph, tau_j_graph, label = 'tau_j')

        #plt.plot(tau_m, label = 'tau_m')
        #plt.plot(tau_h, label = 'tau_h')
        plt.legend()

        plt.figure()
        plt.plot(I_c, label = 'I_c')
        plt.plot(I_p, label = 'I_p')
        plt.plot(I_leak, label = 'I_leak')
        plt.plot(I_Na, label = 'I_Na',ls = '-.')
        plt.legend()


        plt.figure()
        #plt.plot(I_in, label = 'I_in')
        plt.plot(I_out, label = 'I_out')
        plt.legend()



    return I_out
