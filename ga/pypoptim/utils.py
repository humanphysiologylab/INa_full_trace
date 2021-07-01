import pickle

import numpy as np
import pandas as pd

from pypoptim.helpers import RMSE, value_from_bounds



def generate_organism(genes_dict, genes_m_index):

    genes = [value_from_bounds(gene_params['bounds'], log_scale=gene_params['is_multiplier'])
             if gene_name not in state.index else state[exp_cond_name][gene_name]
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



def run_model_ctypes(S, C, config):

    # prepare

    # model comes form mpi_script
    status = run_model_ctypes.model.run(..., output, ...)

    return status, output


def update_phenotype_state(...):
    ...


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
