import pickle
import os

import numpy as np
import pandas as pd

from pypoptim.helpers import RMSE, value_from_bounds



def generate_organism(genes_dict, genes_m_index):

    genes = [value_from_bounds(gene_params['bounds'], log_scale=gene_params['is_multiplier'])
             for exp_cond_name, exp_cond in genes_dict.items() for gene_name, gene_params in exp_cond.items()]

    genes = pd.Series(data=genes, index=genes_m_index)

    organism = dict(genes=genes,
                    state=None)

    return organism


def init_population(config):

    population = [generate_organism(config['runtime']['genes_dict'],
                                    config['runtime']['m_index'])
                  for _ in range(config['runtime']['n_organisms'])]

    return population


def init_population_from_backup(backup, config):

    genes_dict = config['runtime']['genes_dict']

    assert (len(backup) == config['runtime']['n_organisms'])
    assert (len(backup[0]['genes']) == sum(map(len, genes_dict.values())))
    assert (backup[0]['state'].shape == config['runtime']['states_initial'].shape)

    population = [dict(genes=pd.Series(data=organism['genes'], index=config['runtime']['m_index']),
                       state=None)
                  for organism in backup]

    return population



def run_model_ctypes(C, config):

    #kwargs = {}

    S = config['kwargs']['states']['value']


    df_protocol = config['experimental_conditions']['trace_1']['protocol']
    t = df_protocol.t.values
    v_all = df_protocol.v.values
    #print(C)

    df_initial_state_protocol =  config['experimental_conditions']['trace_1']['initial_state_protocol']
    t0 = df_initial_state_protocol.t.values
    v0 = df_initial_state_protocol.v.values

    output_len = len(t)
    initial_state_len = len(t0)

    initial_state_S = np.zeros((initial_state_len, len(S)))

    print(C)

    # x = C.value.copy()#np.concatenate((C.copy(), [18.,-80.]))
    # print(x)
    if config['kwargs'].get('log', None):
        #print('log')
        x[:4] = np.exp(x[:4])
        x[6:8] = np.exp(x[6:8])
        x[10:12] = np.exp(x[10:12])
        x[14:20] = np.exp(x[14:20])
        x[24:28] = np.exp(x[24:28])
        x[27] = (x[27]-5.5)*50/9#want to rescale [-25,25] -> [1,10] and in logarytmic it would be [0,1]
    #print(x)
    #if config['kwargs'].get('old_log', None):
    #    x[:-10] = np.exp(x[:-10])
    run_model_ctypes.model.run(S.values.copy(), C.values.copy(),
            t0, v0, initial_state_len,
            initial_state_S)#, initial_state_A)

    S_output = np.zeros((output_len, len(S)))
    n_sections = 20
    split_indices = np.linspace(0, len(v_all), n_sections + 1).astype(int)

    for k in range(n_sections):
        start, end = split_indices[k], split_indices[k+1]
        v = v_all[start:end]
        t1 = t[start:end] - t[start]
        len_one_step = split_indices[k+1] - split_indices[k]
        status = run_model_ctypes.model.run(initial_state_S[-1].copy(), C.values.copy(),
                                            t1, v, len_one_step,
                                            S_output[start:end])#, output_A[start:end])

    output = S_output.T[-1]
    return status, output


def update_C_from_genes_current(C, genes_current, exp_cond_name, config):

    legend = config['runtime']['legend']
    genes_dict = config['runtime']['genes_dict']
    constants_dict = config['runtime']['constants_dict']
    constants_dict_current = {**constants_dict['common'],
                              **constants_dict[exp_cond_name]}

    for i, g_name in enumerate(genes_current.index.get_level_values('g_name')):

        if g_name in legend['constants'].index:
            for ecn in ['common', exp_cond_name]:
                if g_name in genes_dict[ecn]:
                    if genes_dict[ecn][g_name]['is_multiplier']:
                        C[g_name] *= genes_current[ecn, g_name]
                    else:
                        C[g_name] = genes_current[ecn, g_name]

    for c_name, c in constants_dict_current.items():
        if c_name in legend['constants'].index:
            C[c_name] = c


def update_phenotype(organism, config):

    organism['phenotype'] = dict()

    legend = config['runtime']['legend']

    for exp_cond_name in config['experimental_conditions']:

        if exp_cond_name == 'common':
            continue

        genes_current = organism['genes'][['common', exp_cond_name]]

        C = legend['constants'].copy()
        update_C_from_genes_current(C, genes_current, exp_cond_name, config)

        status, res = run_model_ctypes(C, config)
        if (status != 2) or np.any(np.isnan(res)):
            return 1

        organism['phenotype'][exp_cond_name] = pd.DataFrame(res.T, columns=config['columns_model'])

    return 0


def update_fitness(organism, config):

    loss = 0

    columns_control = config.get("columns_control", ["V"])
    columns_model = config.get("columns_model", ["V"])

    for exp_cond_name, exp_cond in config['experimental_conditions'].items():

        if exp_cond_name == 'common':
            continue

        phenotype_control = exp_cond['phenotype'][columns_control]
        phenotype_model   = organism['phenotype'][exp_cond_name][columns_model]

        if config['loss'] == 'RMSE':
            loss += RMSE(phenotype_control, phenotype_model)

    organism['fitness'] = -loss
