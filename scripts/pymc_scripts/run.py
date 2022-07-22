from ast import In
import os
import sys
import sklearn.covariance as skcov
import aesara
import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
import pymcmcstat
import pymcmcstat.ParallelMCMC

from datetime import datetime
from fastprogress import fastprogress
fastprogress.printing = lambda: True
sys.path.append('../../pypoptim/mpi_scripts/')
from ina_model import InaModel
from io_utils import collect_results
from functions import find_S, big_loglike_x_4

aesara.config.exception_verbosity = 'high'
from functions import func_model
from classes import AwfulLogLike, StupidMetropolis, StupidDEMetropolisZ

dirname = '../../results/'
# dirname = '../../results/pipette_M1_in/all_iv_weights/'
trace_name = '2020_12_22_0015'
dirname_case = os.path.join(dirname, trace_name)
case = os.listdir(dirname_case)[0]

result = collect_results(case, dirname_case, dump_keys=['best'])
sol_best = result['sol_best']
config = result['config']
bounds = config['runtime']['bounds']
phenotype_best = result['phenotype_best']['trace']

model_dir = '../../src/model_ctypes/test_pipette_M1_in/'
# model_dir = '../../src/model_ctypes/pipette_M1_in'
legend_states = pd.read_csv(os.path.join(model_dir, 'legend_states.csv'), index_col='name').value
legend_algebraic = pd.read_csv(os.path.join(model_dir,'legend_algebraic.csv'), index_col='name').value
legend_constants = pd.read_csv(os.path.join(model_dir,'legend_test.csv'), index_col='name').value

A = legend_algebraic.copy()
C = legend_constants.copy()
S = legend_states.copy()

df_protocol = pd.read_csv('../../data/protocols/protocol_trace.csv')
df_initial_state_protocol = pd.read_csv('../../data/protocols/protocol_initial_state.csv')

filename_so = os.path.join(model_dir, 'ina.so')
filename_so_abs = os.path.abspath(filename_so)
Ina = InaModel(filename_so_abs)

data_name = trace_name + '.csv'
data_dirname = '../../data/real/all_activation/'
data_path = os.path.normpath(os.path.join(data_dirname, data_name))
data = np.array(pd.read_csv(data_path, index_col=0).I_out)

weight = np.array(config['experimental_conditions']['trace']['sample_weight'])
weight_grad = np.array(config['experimental_conditions']['trace']['sample_weight_grad'])

m_index = config['runtime']['m_index']
params_list = list(config['runtime']['m_index'].get_level_values(1))
const_from_sol_best = legend_constants.copy()

ind_true=[]
pass_params = ['v_off',
               'c_p', 'R_f',
#                'x_c_comp', 'x_r_comp', 'alpha',
#                'c_p', 'x_c_comp', 'x_r_comp', 'R_f']#, 'alpha',
#                'a0_m', 'b0_m', 's_m', 'delta_m', 'tau_m_const',
#                'a0_h', 'b0_h', 's_h', 'delta_h', 'tau_h_const',
#                'a0_j', 'b0_j', 's_j', 'delta_j', 'tau_j_const',
#                'v_half_m', 'k_m', 'v_half_h', 'k_h', 
#                'v_rev', 'g_leak', 'tau_z'
              ]

for param in pass_params:
    ind = 'common'
    if param == 'v_off':
        ind = 'trace'
    ind_true.append(np.where(m_index==(ind, param)))
mi = m_index.copy()
m_index = m_index.delete(ind_true)
bounds = np.delete(bounds, ind_true, 0)
sol_best_before = np.delete(sol_best.values, ind_true)

bounds[5][0] = 0.001
bounds[22][0] = 2

up, low = 0.9, 1.1
ina_params = ['a0_m', 'b0_m', 'delta_m', 's_m', 'tau_m_const',
              'a0_h', 'b0_h', 'delta_h', 's_h', 'tau_h_const',
              'a0_j', 'b0_j', 'delta_j', 's_j',  'tau_j_const',
              'v_half_h', 'v_half_m', 'k_h', 'k_m'
             ]

trace = func_model(sol_best_before, m_index, Ina=Ina, const=const_from_sol_best, config=config) 
downsampl = 10
pipette_ind = [7, 107, 207, 407]
len_step = int(len(data)/20)
data_cut_size = int(len_step/downsampl)
n = 19

data_cut  = np.array([np.zeros(data_cut_size) for k in range(n)]) 
delta_data = data - trace

for k in range(19):
    data_cut[k] = np.array(delta_data[(k+1)*len_step:(k+2)*len_step:downsampl])
    for p_ind in pipette_ind:
        data_cut[k][p_ind:p_ind+4] = np.nan
data_cut = np.delete(data_cut, np.where(np.isnan(data_cut)), axis=1)


p = np.shape(data_cut)[1]
delta = 2
print(f'n = {n}, p = {p}, delta = {delta}')

cov_0, lw_a = skcov.ledoit_wolf(data_cut, assume_centered=True)
S_0 = np.dot(data_cut.T, data_cut)
beta_0 = n / (1 - lw_a) + p + 1 - n
alpha_0 = np.mean(S_0.diagonal())
phi_0 = alpha_0*(beta_0 + n- p - 1)*lw_a
        
ina_dict = {}
ina_dict['find_S'] = find_S
ina_dict['data'] = data
ina_dict['m_index'] = m_index
ina_dict['Ina'] = Ina
ina_dict['const'] = const_from_sol_best
ina_dict['config'] = config

nchain = 1
ndraws = 10
nburn = 0
print(f'Draws = {ndraws}, Tunes = {nburn}, Chains = {nchain}')


up, low = 0.8, 1.2
new_bounds = []
for i, [index, param_bounds] in enumerate(zip(m_index.get_level_values(1), bounds)):
    if sol_best_before[i]<0:
        ub, lb = sol_best_before[i]*up, sol_best_before[i]*low
    else:
        lb, ub = sol_best_before[i]*up, sol_best_before[i]*low
    if lb < param_bounds[0]:
        lb = param_bounds[0]*1.01
    if ub > param_bounds[1]:
        ub = param_bounds[1]*0.99
    new_bounds.append([lb, ub])
    
new_bounds = np.array(new_bounds)

all_sols = pd.read_csv('../../data/GA_results/sols/2020_12_22_0015.csv')
del_ind = []
for pp in pass_params:
    del_ind.append(np.where(all_sols.g_name == pp)[0][0])
sols = []
for k in all_sols.columns[2:]:
    sol_best_new = np.delete(all_sols[k].values, del_ind)
    sols.append(sol_best_new)
sols+=[sol_best_before]
len_random_start = nchain - len(sols)
if nchain - len(sols)<0:
    len_random_start = 0

initial_values_parameters = pymcmcstat.ParallelMCMC.generate_initial_values(len_random_start,
                                                                 len(new_bounds),
                                                                 new_bounds.T[0],
                                                                 new_bounds.T[1])
initial_values_beta = pymcmcstat.ParallelMCMC.generate_initial_values(nchain,
                                                                      1,
                                                                      9000,
                                                                      11000)
initial_values_parameters = np.append(initial_values_parameters, sols, axis=0)
start_dicts = []
for sol, beta in zip(initial_values_parameters, initial_values_beta):
    start_val = np.array(sol)
    start_vals = {}
    start_vals['parameters'] = start_val
    start_vals['beta'] = np.array(*beta)
    start_dicts.append(start_vals)
start_dicts = np.array(start_dicts)
print(f'start_dicts = {start_dicts}')

param_cov_mat = np.load('long_dangerous_good_cov.npy')

model = pm.Model()
beta_l, beta_u = 1000., 200005.
transform = None

with model:

    beta = pm.Uniform('beta',
                      lower = beta_l,  
                      upper = beta_u,
                      transform = transform)
    
    parameters = pm.Uniform('parameters',
                      lower = bounds.T[0],  
                      upper = bounds.T[1],
                      transform = transform)


    loglike = AwfulLogLike(ina_dict=ina_dict,
                            loglike=big_loglike_x_4,
                            n=n,
                            p=p,
                            delta=delta,
                            S=S_0,
                            phi=(np.diag(np.diag(cov_0)))*11000,
                            cov_mat=cov_0,
                            )

    params = at.as_tensor_variable(parameters,)
    logl_mu = loglike(beta, params)
    pm.Potential("likelihood", logl_mu)

    scale = 1.
    step1 = StupidMetropolis([beta], 
                             loglike = loglike, 
                             bounds = bounds.T, 
                             beta_bounds = [beta_l, beta_u],
                             transform = transform)
    step2 = StupidDEMetropolisZ([parameters],
                             lamb = 0.0, 
                             S = param_cov_mat, 
                             proposal_dist=pm.MultivariateNormalProposal,
                             scaling=scale,
                             tune_interval = 1000,
                             tune = 'S',
                             bounds = bounds.T, 
                             initial_values_size = 28,
                             transform = transform)
                            
    steps = pm.CompoundStep([step2, step1])

idata = pm.sample(ndraws,
                  tune=nburn,
                  step=steps,
                  model = model,
                  chains=nchain,
                  cores=nchain,
                  return_inferencedata=True,
                  initvals=start_dicts,
                 )

fold = '../../results/postprocessing/pymc/AwfullUpgraded/test/'
# fold = f'../../results/postprocessing/pymc/AwfullUpgraded/long_awfulLoglike_x_4_start_values/{nchain}_chain_{ndraws}_draws_scale_{scale}_tune_S_covmat_from_Roman'

foldername = os.path.join(fold, trace_name)

fold_list = foldername.split('/')
for k in range(2, len(fold_list)+1):
    loc_path = os.path.join(*foldername.split('/')[:k])
    logic = os.path.isdir(loc_path)
    if not logic:
        os.mkdir(loc_path)

time_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
filename_last = os.path.join(foldername, time_suffix+'.nc')
idata.to_netcdf(filename_last)
print("ALL DONE")
