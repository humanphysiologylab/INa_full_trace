#!/usr/bin/env python
import copy
import os
import logging
import argparse
import warnings

from mpi4py import MPI
from tqdm.auto import tqdm

import numpy as np

from ina_model import InaModel
from solmodel import SolModel
from pypoptim.algorythm.ga import GA

from pypoptim.helpers import argmin, is_values_inside_bounds
from pypoptim import Timer

from io_utils import prepare_config, update_output_dict, backup_config, dump_epoch, save_sol_best
from mpi_utils import allocate_recvbuf, allgather, population_from_recvbuf
from pypoptim.losses import RMSE


def mpi_script(config_filename):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    config = None
    if comm_rank == 0:
        config = prepare_config(config_filename)
        config['runtime']['comm_size'] = comm_size

        print(f"# commit: {config['runtime']['sha']}")
        print(f'# size: {comm_size}')

        if config['n_organisms'] % comm_size != 0:
            config['runtime']['n_organisms'] = int(np.ceil(config['n_organisms'] / comm_size) * comm_size)
            print(f'# n_organisms: {config["n_organisms"]} to {config["runtime"]["n_organisms"]}',
                  flush=True)
        else:
            config['runtime']['n_organisms'] = config['n_organisms']

        update_output_dict(config)
        os.makedirs(config['runtime']['output']['folder'])
        print(f"# folder: {config['runtime']['output']['folder']}", flush=True)

    config = comm.bcast(config, root=0)

    recvbuf_dict = allocate_recvbuf(config, comm)

    model = InaModel(config['runtime']['filename_so_abs'])
    SolModel.model = model
    SolModel.config = config

    rng = np.random.Generator(np.random.PCG64(42 + comm_rank))
    warnings.warn("УБЕРИ СИД!!!!")

    ga_optim = GA(SolModel,
                  bounds=config['runtime']['bounds'],
                  gammas=config['runtime']['gammas'],
                  mask_log10_scale=config['runtime']['mask_multipliers'],
                  rng=rng)

    initial_population_filename = config.get('initial_population_filename', None)
    if initial_population_filename is not None:
        raise NotImplementedError

    if comm_rank == 0:
        backup_config(config)

    batch = ga_optim.generate_population(config['runtime']['n_orgsnisms_per_process'])

    timer = Timer()

    if comm_rank == 0:
        pbar = tqdm(total=config['n_generations'], ascii=True)

    loss_last = np.inf

    dirname_save = os.path.join(config['runtime']['output']['folder'], 'phenotypes_all', str(comm_rank))
    os.makedirs(dirname_save, exist_ok=True)

    for epoch in range(config['n_generations']):

        timer.start('calc')
        if comm_rank == 0:
            pbar.set_postfix_str("CALC")
        for i, sol in enumerate(batch):
            y_previous = None if not sol.is_updated() else sol.y.copy()
            sol.update()

            if not (sol.is_valid() and ga_optim.is_solution_inside_bounds(sol)):
                sol._y = np.inf
            else:
                for exp_cond_name in config['experimental_conditions']:
                    if exp_cond_name == 'common':
                        continue
                    filename_save = f"{epoch:03d}_{i:03d}_{exp_cond_name}.npy"
                    filename_save = os.path.join(dirname_save, filename_save)
                    np.save(filename_save, sol['phenotype'][exp_cond_name].values)


        comm.Barrier()
        timer.end('calc')

        timer.start('gather')
        if comm_rank == 0:
            pbar.set_postfix_str("GATHER")
        allgather(batch, recvbuf_dict, comm)
        population = population_from_recvbuf(recvbuf_dict, SolModel, config)

        n_orgsnisms_per_process = config['runtime']['n_orgsnisms_per_process']
        shift = comm_rank * n_orgsnisms_per_process
       # assert all(sol_b.is_all_equal(sol_p) for sol_b, sol_p in zip(batch, population[shift:]))

        timer.end('gather')

        timer.start('save')
        if comm_rank == 0:
            pbar.set_postfix_str("SAVE")

        index_best = argmin(population)
        assert population[index_best] is min(population)
        comm_rank_best = index_best // config['runtime']['n_orgsnisms_per_process']
        index_best_batch = index_best % config['runtime']['n_orgsnisms_per_process']

        loss_current = min(population).y
        #assert loss_current <= loss_last
        loss_current = loss_last

        if comm_rank == comm_rank_best:
            # print(batch)
            sol_best = batch[index_best_batch]

            assert sol_best is min(batch)
            assert sol_best.is_all_equal(min(population))

            save_sol_best(sol_best, config)

            assert sol_best.is_updated()
            assert sol_best.is_valid()
            assert is_values_inside_bounds(sol_best.x, config['runtime']['bounds'])
            assert ga_optim.is_solution_inside_bounds(sol_best)

        if comm_rank == (comm_rank_best + 1) % comm_size:
            dump_epoch(recvbuf_dict, config)
        timer.end('save')

        timer.start('gene')
        if comm_rank == 0:
            pbar.set_postfix_str("GENE")

        population = ga_optim.filter_population(population)
        # n_invalids = config['runtime']['n_organisms'] - len(population)
        # percentage_invalid = n_invalids / config['runtime']['n_organisms'] * 100
        population.sort()

        if len(population) <= 3:
            if comm_rank == 0:
                msg = f"# Not enough organisms for genetic operations left: {len(population)}"
                raise RuntimeError(msg)

        elites_all = population[:config['n_elites']]  # len may be less than config['n_elites'] due to invalids

        elites_batch = []
        for sol_elite in elites_all:
            for sol in batch:
                if sol.is_all_equal(sol_elite):
                    elites_batch.append(sol)

        for sol in batch:
            #sol_copy = SolModel(sol.x.copy())
            #sol_copy._y = sol.y
            sol.update()
           # assert SolModel.config == sol.config
            #if sol.y != sol_copy.y:
               # print(len(sol.data['phenotype']['trace']['I_out']), len(sol_copy.data['phenotype']['trace']['I_out']))
               # print(RMSE(sol.data['phenotype']['trace']['I_out'], sol_copy.data['phenotype']['trace']['I_out']))
               # print("Y \n", sol.y,"sol_y\n", sol_copy.y,"sol_y_copy\n",sol.x,"sol_x\n", sol_copy.x,"sol_x_copy\n END")

                #assert 0
           # if np.all(sol.x != sol_copy.x):
                #print("X \n", sol.y, "\n", sol_copy.y, "\n", sol.x, "\n", sol_copy.x)
                #assert 0

        # elites_batch = elites_all[comm_rank::comm_size]  # elites_batch may be empty
        n_elites = len(elites_batch)
        assert n_elites <= len(batch)
        n_mutants = config['runtime']['n_orgsnisms_per_process'] - n_elites

        mutants_batch = ga_optim.get_mutants(population, n_mutants)
        batch = elites_batch + mutants_batch

        assert (len(batch) == config['runtime']['n_orgsnisms_per_process'])

        timer.end('gene')

        if comm_rank == 0:
            with open(os.path.join(config['runtime']['output']['folder'], 'runtime.log'), 'w') as f:
                print(timer.report(), file=f)
                print(f'# epoch: {epoch}', file=f)
            pbar.update(1)
            pbar.refresh()

        timer.clear()

    if comm_rank == 0:
        pbar.set_postfix_str("DONE")
        pbar.refresh()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='configuration file')
    parser.add_argument("-v", "--verbosity",
                        type=str,
                        help="logging level")

    args = parser.parse_args()

    config_filename = args.config
    logging_level = args.verbosity
    level = dict(INFO=logging.INFO,
                 DEBUG=logging.DEBUG).get(logging_level, logging.WARNING)
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)
    logging.getLogger('numba').setLevel(logging.CRITICAL)  # https://stackoverflow.com/a/63471108/13213091

    mpi_script(config_filename)
