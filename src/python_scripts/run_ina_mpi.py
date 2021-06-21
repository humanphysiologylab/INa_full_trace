import numpy as np
import pandas as pd
import scipy.optimize as scop
import os
import ctypes
import sys
from sklearn.metrics import mean_squared_error as MSE
#from cmaes import CMA
#import MPIpool
from mpipool import MPIPool
from mpi4py import MPI
import time
import csv



#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

#print(f'{rank} / {size}')

sys.path.append("../src/python_scripts/")
from functions import scale, rescale, calculate_full_trace, give_me_ina,loss

dirname = '../model_ctypes/ina/'

name = 'result_test_ina.csv'
file_to_write = (f"../../data/results/{name}")

legend_constants = pd.read_csv(os.path.join(dirname, "legend_constants.csv"), index_col='name')
legend_states = pd.read_csv(os.path.join(dirname, "legend_states.csv"), index_col='name')['value']
legend_algebraic = pd.read_csv(os.path.join(dirname, "legend_algebraic.csv"), index_col='name')['value']

S = legend_states.copy()
R = S.copy() * 0
C = legend_constants.copy()
A = legend_algebraic.copy()

filename_protocol = "../../data/protocols/protocol_79.csv"
df_protocol = pd.read_csv(filename_protocol)

diff = (df_protocol.v.shift() - df_protocol.v).fillna(0)
mask_capacity = diff.rolling(window=200).sum() !=0
indices_weight_c = mask_capacity.index.to_numpy()[mask_capacity] + 15

mask_INa = diff.rolling(window=600).sum()<-10
indices_weight_INa = mask_INa.index.to_numpy()[mask_INa] + 100

weight = np.zeros_like(df_protocol.v) + 5
weight[indices_weight_c] = 15
weight[indices_weight_INa] = 60

bounds = np.array((C.bound_1.values[:-2],C.bound_2.values[:-2]))
scale_bounds = np.array([[0.,1.] for k in range(len(bounds.T))])
##KWARGS WORKING
t0 = np.arange(0, 0.25, 5e-5)
v0 = np.full_like(t0, -80.0)
initial_state_len = len(t0)

t = df_protocol['t'].values
v = df_protocol['v'].values
output_len = len(t)

initial_state_S = pd.DataFrame(np.zeros((initial_state_len, len(S))), columns=legend_states.index)
initial_state_A = pd.DataFrame(np.zeros((initial_state_len, len(A))), columns=legend_algebraic.index)

output_S = pd.DataFrame(np.zeros((output_len, len(S))), columns=legend_states.index)
output_A = pd.DataFrame(np.zeros((output_len, len(A))), columns=legend_algebraic.index)



filename = 'ina.so'
filename_so = os.path.join(dirname, filename)
filename_abs = os.path.abspath(filename_so)
kwargs = dict(S = S,
              t0 = t0,
              v0 = v0,
              initial_state_S = initial_state_S,
              initial_state_A = initial_state_A,
              initial_state_len = initial_state_len,
              t = df_protocol['t'].values,
              v = df_protocol['v'].values,
              filename_abs = filename_abs,
              output_S = output_S,
              output_A = output_A,
              bounds = bounds,
              sample_weight = weight,
             )

data = pd.read_csv('../../data/training/2020_12_19_0035 I-V INa 11,65 pF.atf' ,delimiter= '\t', header=None, skiprows = 11)
exp_data = np.concatenate([data[k] for k in range(1,21)])

x0 = scale(C.value.values[:-2],*bounds)

#start_time = time.process_time ()
data = calculate_full_trace(x0, kwargs)

#with open(file_to_write, "w", newline='') as csv_file:
#    writer = csv.writer(csv_file, delimiter=',')
#    writer.writerow(('generation',*C[:-2].T.columns,'loss'))

result = 0
print('start')
with MPIPool() as pool:
    pool.workers_exit()
    #exit()
    result = scop.differential_evolution(loss,
                                    bounds=scale_bounds,
                                    args=(data, kwargs),
                                    maxiter=10,
                                    disp = True,
                                    updating='deferred',
                                    popsize = 10,
                                    workers = pool.map,
                                    #callback = print_fun_DE,
                                    #disp = True,
                                    seed=42)

    print('RESULT = ',result)


##### CMAES
#if rank == 0:

#    mean = x0 + 0.001
#    sigma = 0.5
#    population_size_per_process = 1000
#    population_size = population_size_per_process * size
#    optimizer = CMA(mean=mean, sigma=sigma, bounds=scale_bounds, seed=0, population_size=population_size)



#for generation in range(1000):
#    if rank == 0:
#        x_list = [[optimizer.ask() for _ in range(population_size_per_process)] for _ in range(size)]
#    else:
#        x_list = None

#    x_list = comm.scatter(x_list, root=0)
#    solutions = []

#    for x in x_list:
#        value = loss(x, data, kwargs)
#        solutions.append((x, value))

#    solutions = comm.gather(solutions, root=0)

#    if rank == 0:
#        solutions = sum(solutions, [])
#        with open(file_to_write, "a", newline='') as csv_file:
#            for sol in solutions:
#                writer = csv.writer(csv_file, delimiter=',')
#                writer.writerow((generation, *sol[0], sol[1]))
#        optimizer.tell(solutions)


print('OK')
