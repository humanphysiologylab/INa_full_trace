import numpy as np
import pandas as pd
import sys
import os
import scipy.optimize as scop
import json

sys.path.append('pypoptim/mpi_scripts/')

from loss_utils import calculate_loss
from io_utils import collect_results
from ina_model import InaModel
from solmodel import SolModel


def return_configs(m_dirname, m_dirs):
    sols = []
    configs = []
    keyword = 'trace'
    for dirname in m_dirs:
        dirname_results = os.path.normpath(os.path.join(m_dirname, dirname)) 
        cases = os.listdir(dirname_results)
        for case in cases:
            result = collect_results(case, dirname_results, dump_keys=['best', 'dump'])
            sol_best = result['sol_best'].copy()
            sols.append(sol_best)
            config = result['config'].copy()
            configs.append(config)

    return  sols, configs

history = []
def callback(x):
    history.append(loss_func(x))

maxiter = 1000
m1_dirname = 'results/M3_real/'
m1_dirs = os.listdir(m1_dirname)
m1_sols, m1_configs = return_configs(m1_dirname, m1_dirs)

filename_res = 'res_NM.json'
filename_loss = 'loss_NM.csv'


config_start = m1_configs[0]
filename_so = config_start['runtime']['filename_so_abs']
model = InaModel(filename_so)
SolModel.model = model

def loss_func(x, *args):
    sol = SolModel(x)
    sol.update()
    return sol.y
print("Starting Nelder-Mead optimization")
for sol, config, dirname in zip(m1_sols, m1_configs, m1_dirs):
    dirname_results = os.path.normpath(os.path.join(m1_dirname, dirname))
    cases = os.listdir(dirname_results)
    for case in cases:
        print("Case = ",dirname_results, case)
        SolModel.config = config.copy()
        res = scop.minimize(loss_func, sol, 
                            method = 'Nelder-Mead', bounds = config['runtime']['bounds'], 
                            options={'maxiter':maxiter}, callback=callback)

        res.final_simplex = res.final_simplex[0].tolist()
        res.x = res.x.tolist()

        full_filename_res = os.path.normpath(os.path.join(dirname_results,case,filename_res))
#         print(full_filename_res)


        with open(full_filename_res, 'w') as fp:
            json.dump(res, fp)

        full_filename_loss = os.path.normpath(os.path.join(dirname_results,case,filename_loss))
        np.savetxt(full_filename_loss, history, delimiter=',')
        history = []
