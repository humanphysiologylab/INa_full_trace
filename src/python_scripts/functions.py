import numpy as np
import ctypes
from sklearn.metrics import mean_squared_error as MSE

#####################################################################

def scale(x, min_bound, max_bound):
    y = (x.copy() - min_bound)/(max_bound -  min_bound)
    return(y)

#####################################################################

def rescale(x, min_bound, max_bound):
    y = x.copy() * (max_bound - min_bound) + min_bound
    return(y)

#####################################################################

def calculate_full_trace(y, *args):

    kwargs = args[-1]

    t = kwargs['t']#there should be time for the first step
    v_all = kwargs['v']

    output_S = kwargs['output_S'].values.copy()
    output_A = kwargs['output_A'].values.copy()
    bounds = kwargs['bounds']

    S = kwargs['S']

    t0 = kwargs['t0']
    v0 = kwargs['v0']

    initial_state_S = kwargs['initial_state_S'].values.copy()
    initial_state_A = kwargs['initial_state_A'].values.copy()
    initial_state_len = kwargs['initial_state_len']
    filename_abs = kwargs['filename_abs']


    if kwargs.get('rescale'):
        #print('rescale')
        x = np.concatenate((rescale(y.copy(), *bounds), [18.,-80.]))

    else:
        #print('Oh NOOOOO, my dear, there no rescale at all!!!')
        #x = np.concatenate((y.copy(), [18.,-80.]))
        x = np.concatenate((y.copy(), [18.,-80.]))
    if kwargs.get('log', None):
        #print('log')
        x[:4] = np.exp(x[:4])
        x[6:8] = np.exp(x[6:8])
        x[10:12] = np.exp(x[10:12])
        x[14:20] = np.exp(x[14:20])
        x[24:28] = np.exp(x[24:28])
        x[27] = (x[27]-5.5)*50/9#want to rescale [-25,25] -> [1,10] and in logarytmic it would be [0,1]
    #print(x)
    ina = give_me_ina(filename_abs)
    ina.run(S.values.copy(), x,
            t0, v0, initial_state_len,
            initial_state_S, initial_state_A)

    S0 = initial_state_S[-1]
    #print(S0)
    n_sections = 20
    split_indices = np.linspace(0, len(v_all), n_sections + 1).astype(int)

    for k in range(n_sections):
        start, end = split_indices[k], split_indices[k+1]
        v = v_all[start:end]
        t1 = t[start:end] - t[start]
        len_one_step = split_indices[k+1] - split_indices[k]
        status = ina.run(S0.copy(), x.copy(),
                         t1, v, len_one_step,
                         output_S[start:end], output_A[start:end])

    I_out = output_S.T[-1]
    #I_out = output_S.I_out.copy()
    return I_out

#####################################################################

def OLD_calculate_full_trace(y, *args):
    kwargs = args[-1]

    t = kwargs['t']#there should be time for the first step
    v_all = kwargs['v']
    dt = kwargs['dt']
    output_S = kwargs['output_S'].values.copy()
    output_A = kwargs['output_A'].values.copy()
    bounds = kwargs['bounds']

    S = kwargs['S']

    t0 = kwargs['t0']
    v0 = kwargs['v0']

    initial_state_S = kwargs['initial_state_S'].values.copy()
    initial_state_A = kwargs['initial_state_A'].values.copy()
    initial_state_len = kwargs['initial_state_len']
    filename_abs = kwargs['filename_abs']
    if kwargs.get('rescale'):
        #print('rescale')
        x = np.concatenate((rescale(y.copy(), *bounds), [18.,-80.]))
    else:
        #print('Oh NOOOOO, my dear, there no rescale at all!!!')
        x = np.concatenate((y.copy(), [18.,-80.]))

    ina = OLD_give_me_ina(filename_abs)
    ina.run(dt, S.values.copy(), x,
            t0, v0, initial_state_len,
            #initial_state_S.values, initial_state_A.values)
            initial_state_S, initial_state_A)

    #S0 = initial_state_S.values[-1]
    S0 = initial_state_S[-1]


    n_sections = 20
    split_indices = np.linspace(0, len(v_all), n_sections + 1).astype(int)

    for k in range(n_sections):
        start, end = split_indices[k], split_indices[k+1]
        v = v_all[start:end]
        t1 = t[start:end] - t[start]
        len_one_step = split_indices[k+1] - split_indices[k]
        status = ina.run(dt,S0.copy(), x.copy(),
                         t1, v, len_one_step,
                         #output_S.values[start:end], output_A.values[start:end])
                        output_S[start:end], output_A[start:end])

    #I_out = output_S.I_out.copy()
    I_out = output_S.T[-1]
    #I_out = initial_state_S.I_out.copy()

    return I_out

#####################################################################


def give_me_ina(filename):

    ina = ctypes.CDLL(filename)


    # void initialize_states_default(double *STATES)
    ina.initialize_states_default.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.initialize_states_default.restype = ctypes.c_void_p


    # void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES)
    ina.compute_rates.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_rates.restype = ctypes.c_void_p


    # void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC)
    ina.compute_algebraic.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_algebraic.restype = ctypes.c_void_p


    # int run(double *S, double *C,
    #         double *time_array, double *voltage_command_array, int array_length,
    #         double *output_S, double *output_A)
    ina.run.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ]
    ina.run.restype = ctypes.c_int
    return ina
#####################################################################

def OLD_give_me_ina(filename):

    ina = ctypes.CDLL(filename)


    # void initialize_states_default(double *STATES)
    ina.initialize_states_default.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.initialize_states_default.restype = ctypes.c_void_p


    # void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES)
    ina.compute_rates.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_rates.restype = ctypes.c_void_p


    # void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC)
    ina.compute_algebraic.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    ina.compute_algebraic.restype = ctypes.c_void_p


    # int run(double *S, double *C,
    #         double *time_array, double *voltage_command_array, int array_length,
    #         double *output_S, double *output_A)
    ina.run.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ]
    ina.run.restype = ctypes.c_int
    return ina

#####################################################################

def loss(y, *args):
    #return 42
    kwargs = args[-1]
    data = args[0]
    sample_weight = kwargs.get('sample_weight', None)
    I_out = calculate_full_trace(y, *args)
    if np.any(np.isnan(I_out)):
        return np.inf
    if np.any(np.isinf(I_out)):
        return np.inf
    return MSE(data, I_out, sample_weight=sample_weight)

#####################################################################

def OLD_loss(y, *args):
    #return 42
    kwargs = args[-1]
    data = args[0]
    sample_weight = kwargs.get('sample_weight', None)
    I_out = OLD_calculate_full_trace(y, *args)

    if np.any(np.isnan(I_out)):
        return np.inf
    if np.any(np.isinf(I_out)):
        return np.inf
    return MSE(data, I_out, sample_weight=sample_weight)

######################################################################

def spec_log_scale(x):
    y = x.copy()
    y[:4] = np.log(y[:4])
    y[6:8] = np.log(y[6:8])
    y[10:12] = np.log(y[10:12])
    y[14:20] = np.log(y[14:20])
    y[24:27] = np.log(y[24:27])
    y[27] = np.log(9*y[27]/50 + 5.5)
    return y
