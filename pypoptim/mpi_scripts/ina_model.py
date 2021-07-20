import os
import ctypes
import pandas as pd
import numpy as np

class InaModel:

    def __init__(self, filename_so):

        filename_so_abs = os.path.abspath(filename_so)
        ctypes_obj = ctypes.CDLL(filename_so_abs)

        # void initialize_states_default(double *STATES)
        ctypes_obj.initialize_states_default.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]
        ctypes_obj.initialize_states_default.restype = ctypes.c_void_p
        ctypes_obj.compute_rates.argtypes = [
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]
        ctypes_obj.compute_rates.restype = ctypes.c_void_p
        ctypes_obj.compute_algebraic.argtypes = [ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        ]
        ctypes_obj.compute_algebraic.restype = ctypes.c_void_p
        ctypes_obj.run.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        ]
        ctypes_obj.run.restype = ctypes.c_int

        self._run = ctypes_obj.run
        self._status = None

    @property
    def status(self):
        return self._status

    def run(self, S, C, **kwargs):
        # def run_model_ctypes(C, config):

        #kwargs = kwargs['kwargs']

        # S = kwargs['states']['value']
        A = kwargs['runtime']['algebraic']
        # print(A)


        df_protocol = kwargs['runtime']['protocol']
        #print(df_protocol)
        t = df_protocol.t.values
        v_all = df_protocol.v.values

        df_initial_state_protocol = kwargs['runtime']['initial_protocol']
        t0 = df_initial_state_protocol.t.values
        v0 = df_initial_state_protocol.v.values

        output_len = len(t)
        initial_state_len = len(t0)

        initial_state_S = np.zeros((initial_state_len, len(S)))
        initial_state_A = np.zeros((initial_state_len, len(A)))

        # x = C.value.copy()
        # np.concatenate((C.copy(), [18.,-80.]))
        self._run(S.values.copy(), C.values.copy(),
                t0, v0, initial_state_len,
                initial_state_S, initial_state_A)

        S_output = np.zeros((output_len, len(S)))
        A_output = np.zeros((output_len, len(A)))
        n_sections = 20
        split_indices = np.linspace(0, len(v_all), n_sections + 1).astype(int)

        for k in range(n_sections):
            start, end = split_indices[k], split_indices[k + 1]
            v = v_all[start:end]
            t1 = t[start:end] - t[start]
            len_one_step = split_indices[k + 1] - split_indices[k]
            status = self._run(initial_state_S[-1].copy(), C.values.copy(),
                           t1, v, len_one_step,
                           S_output[start:end], A_output[start:end])
        #output = pd.DataFrame(np.empty((n_samples_per_stim * n_beats + 1, len(S))),
        #                      columns=S.index)

        output = S_output.T[-1]
        return output
