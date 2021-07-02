from mpi4py import MPI
import os
import json
import time
import pickle
from datetime import datetime
import subprocess

import numpy as np
import pandas as pd
import sys

import ctypes
import git

from pypoptim.helpers import batches_from_list, argmax_list_of_dicts, \
                             Timer, find_index_first, strip_comments, \
                             DoubleArrayType_1D, DoubleArrayType_2D

from pypoptim.algorythm.ga import do_step, transform_genes_bounds, transform_genes_bounds_back

from pypoptim.cardio import create_genes_dict_from_config, create_constants_dict_from_config, \
                            generate_bounds_gammas_mask_multipliers, \
                            save_epoch

from utils import init_population, init_population_from_backup, \
                  run_model_ctypes, \
                  update_phenotype, update_fitness

#### ##    ## #### ########
 ##  ###   ##  ##     ##
 ##  ####  ##  ##     ##
 ##  ## ## ##  ##     ##
 ##  ##  ####  ##     ##
 ##  ##   ###  ##     ##
#### ##    ## ####    ##

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if len(sys.argv) < 2:
    if comm_rank == 0:
        print(f"Usage: mpiexec -n 2 python {sys.argv[0]} config.json")
    exit()

config_filename = sys.argv[1]
config_path = os.path.dirname(os.path.realpath(config_filename))

with open(config_filename) as f:
    text = f.read()
    text = strip_comments(text)
    config = json.loads(text)

config['runtime'] = dict()

config['runtime']['sha'] = git.Repo(search_parent_directories=True).head.commit.hexsha

config['runtime']['genes_dict'] = create_genes_dict_from_config(config)
config['runtime']['constants_dict'] = create_constants_dict_from_config(config)

m_index_tuples = [(exp_cond_name, gene_name) for exp_cond_name, gene in config['runtime']['genes_dict'].items() for gene_name in gene]
m_index = pd.MultiIndex.from_tuples(m_index_tuples)
m_index.names = ['ec_name', 'g_name']

config['runtime']['m_index'] = m_index


 ######  ######## ##    ## ########  ########  ######
##    ##    ##     ##  ##  ##     ## ##       ##    ##
##          ##      ####   ##     ## ##       ##
##          ##       ##    ########  ######    ######
##          ##       ##    ##        ##             ##
##    ##    ##       ##    ##        ##       ##    ##
 ######     ##       ##    ##        ########  ######

filename_so = os.path.join(config_path, config["filename_so"])
filename_so_abs = os.path.abspath(filename_so)

dirname_so = os.path.split(filename_so_abs)[0]
for rule in ['clean'], []:
    popenargs = ["make"] + rule + ["-C", dirname_so]
    output = subprocess.check_output(popenargs)
    print(output.decode())

model = ctypes.CDLL(filename_so_abs)

    # void initialize_states_default(double *STATES)
model.initialize_states_default.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
]
model.initialize_states_default.restype = ctypes.c_void_p


# void compute_rates(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC, double *RATES)
model.compute_rates.argtypes = [
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
]
model.compute_rates.restype = ctypes.c_void_p


# void compute_algebraic(const double time,  double *STATES, double *CONSTANTS,  double *ALGEBRAIC)
model.compute_algebraic.argtypes = [
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
]
model.compute_algebraic.restype = ctypes.c_void_p


# int run(double *S, double *C,
#         double *time_array, double *voltage_command_array, int array_length,
#         double *output_S, double *output_A)
model.run.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    #np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
]
model.run.restype = ctypes.c_int

# create a new attribute of the function
run_model_ctypes.model = model  # <- will be used later

###############################################################################


legend = dict()
legend['states'] = pd.read_csv(os.path.join(config_path, config["filename_legend_states"]),
                               usecols=['name', 'value'], index_col='name')['value']  # Series
legend['constants'] = pd.read_csv(os.path.join(config_path, config["filename_legend_constants"]),
                                  usecols=['name', 'value'], index_col='name')['value']  # Series
config['runtime']['legend'] = legend


for exp_cond_name, exp_cond in config['experimental_conditions'].items():

    if exp_cond_name == 'common':
        continue

    filename_phenotype = os.path.normpath(os.path.join(config_path, exp_cond['filename_phenotype']))
    exp_cond['phenotype'] = pd.read_csv(filename_phenotype)
    exp_cond['filename_phenotype'] = filename_phenotype

    filename_protocol = os.path.normpath(os.path.join(config_path, exp_cond['filename_protocol']))
    exp_cond['protocol'] = pd.read_csv(filename_protocol)
    exp_cond['filename_protocol'] = filename_protocol

    filename_initial_state_protocol = os.path.normpath(os.path.join(config_path, exp_cond['filename_initial_state_protocol']))
    exp_cond['initial_state_protocol'] =  pd.read_csv(filename_initial_state_protocol)
    exp_cond['filename_initial_state_protocol'] = filename_protocol


filename_kwargs = os.path.normpath(os.path.join(config_path, config['kwargs']['filename_legend_states']))
config['kwargs']['states'] =  pd.read_csv(filename_kwargs, index_col='name')
config['kwargs']['filename_legend_states'] = filename_kwargs


output_folder_name = os.path.normpath(os.path.join(config_path, config.get("output_folder_name", "./output")))
if comm_rank == 0:
    time_suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    output_folder_name = os.path.join(output_folder_name, time_suffix)
    print(f"# output_folder_name was set to {output_folder_name}", flush=True)
    config['runtime']['time_suffix'] = time_suffix
else:
    output_folder_name = None
output_folder_name = comm.bcast(output_folder_name, root=0)  # due to time call
config['runtime']["output_folder_name"] = output_folder_name
comm.Barrier()

config['runtime']['output'] = dict(output_folder_name_phenotype = os.path.join(output_folder_name, "phenotype"),
                                   dump_filename                = os.path.join(output_folder_name, "dump.bin"),
                                   dump_last_filename           = os.path.join(output_folder_name, "dump_last.npy"),
                                   dump_elite_filename          = os.path.join(output_folder_name, "dump_elite.npy"),
                                   backup_filename              = os.path.join(output_folder_name, "backup.pickle"),
                                   config_backup_filename       = os.path.join(output_folder_name, "config_backup.pickle"),
                                   log_filename                 = os.path.join(output_folder_name, "runtime.log"),
                                   organism_best_filename       = os.path.join(output_folder_name, "organism_best.pickle"),
                                   genes_best_filename          = os.path.join(output_folder_name, "genes_best.csv"))

if comm_rank == 0:
    for folder in config['runtime']['output']['output_folder_name_phenotype'],:
        os.makedirs(folder, exist_ok=True)
    with open(config['runtime']['output']['dump_filename'], "wb") as file_dump:  # create or clear and close
        pass
    with open(config['runtime']['output']['dump_elite_filename'], "wb") as f:  # create or clear and close
        pass
    with open(config['runtime']['output']['log_filename'], "w") as file_log:
        file_log.write(f"# SIZE = {comm_size}\n")
        file_log.write(f"# commit {config['runtime']['sha']}\n")

time_start = time.time()

bounds, gammas, mask_multipliers = generate_bounds_gammas_mask_multipliers(config['runtime']['genes_dict'])
config['runtime']['bounds'] = bounds
config['runtime']['gammas'] = gammas
config['runtime']['mask_multipliers'] = mask_multipliers

config['runtime']['kw_ga'] = dict(crossover_rate=config.get('crossover_rate', 1.0),
                                  mutation_rate=config.get('mutation_rate', 0.1),
                                  gamma=config.get('gamma', 1.0))


if config['n_organisms'] % comm_size != 0:
    config['runtime']['n_organisms'] = int(np.ceil(config['n_organisms'] / comm_size) * comm_size)
    if comm_rank == 0:
        print(f'# `n_organisms` is changed from {config["n_organisms"]} to {config["runtime"]["n_organisms"]}', flush=True)
else:
    config['runtime']['n_organisms'] = config['n_organisms']


n_orgsnisms_per_process = config['runtime']['n_organisms'] // comm_size

genes_size = sum(map(len, config['runtime']['genes_dict'].values()))

recvbuf_genes = np.empty([comm_size, n_orgsnisms_per_process * genes_size])
recvbuf_fitness = np.empty([comm_size, n_orgsnisms_per_process * 1])


if comm_rank == 0:
    initial_population_filename = config.get('initial_population_filename', None)
    if initial_population_filename is not None:
        initial_population_filename = os.path.normpath(os.path.join(config_path, initial_population_filename))
        with open(initial_population_filename, 'rb') as f:
            backup = pickle.load(f)
        population = init_population_from_backup(backup, config)
        config['runtime']['initial_population_filename'] = initial_population_filename
        with open(config['runtime']['output']['log_filename'], "a") as file_log:
            file_log.write(f"population was loaded from {initial_population_filename}\n")
    else:
        population = init_population(config)

    population = batches_from_list(population, comm_size)
else:
    population = None

batch = comm.scatter(population, root=0)

if comm_rank == 0:
    with open(config['runtime']['output']['config_backup_filename'], "wb") as file_config_backup:
        pickle.dump(config, file_config_backup)

timer = Timer()

##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

for epoch in range(config['n_generations']):

    #  calculations
    timer.start('calc')

    index_best_per_batch = 0  # index of the best organism per batch, used for memory-optimization

    for i, organism in enumerate(batch):

        status = update_phenotype(organism, config)  # TODO

        FLAG_INVALID = False
        msg_invalid = "Invalid organism:\n"
        if status != 0:
            FLAG_INVALID = True
            msg_invalid += f"# status = {status}\n"
        elif np.any(organism['genes'] <= bounds[:, 0]) or np.any(organism['genes'] >= bounds[:, 1]):
            FLAG_INVALID = True
            s = " ".join([f"{'*' if (gene <= bounds[i, 0] or gene >= bounds[i, 1]) else ''}{gene:.3e}"
                          for i, gene in enumerate(organism['genes'])])
            msg_invalid += f"# out of bounds: {s}\n"

        if FLAG_INVALID:
            organism['fitness'] = np.NINF
            print(msg_invalid)
            del organism['phenotype']
        else:
            update_fitness(organism, config)  # TODO
            if organism['fitness'] > batch[index_best_per_batch]['fitness']:
                if 'phenotype' in batch[index_best_per_batch]:  # best guy could be invalid
                    del batch[index_best_per_batch]['phenotype']
                index_best_per_batch = i
            elif i:
                del organism['phenotype']
    timer.end('calc')

     ######      ###    ######## ##     ## ######## ########
    ##    ##    ## ##      ##    ##     ## ##       ##     ##
    ##         ##   ##     ##    ##     ## ##       ##     ##
    ##   #### ##     ##    ##    ######### ######   ########
    ##    ##  #########    ##    ##     ## ##       ##   ##
    ##    ##  ##     ##    ##    ##     ## ##       ##    ##
     ######   ##     ##    ##    ##     ## ######## ##     ##

    timer.start('gather_sendbuf')
    sendbuf_genes = np.concatenate([organism['genes'] for organism in batch])
    sendbuf_fitness = np.array([organism['fitness'] for organism in batch])
    assert(not np.any(np.isnan(sendbuf_fitness)))
    timer.end('gather_sendbuf')

    timer.start('gather_allgather')
    comm.Allgatherv(sendbuf_genes, recvbuf_genes)
    comm.Allgatherv(sendbuf_fitness, recvbuf_fitness)
    timer.end('gather_allgather')

    timer.start('gather_recvbuf')
    recvbuf_genes = recvbuf_genes.reshape((config['runtime']['n_organisms'], genes_size))
    recvbuf_fitness = recvbuf_fitness.flatten()
    timer.end('gather_recvbuf')

    timer.start('gather_population')
    #  assert (not np.any(np.isnan(recvbuf_fitness)))

    population = [dict(genes     = recvbuf_genes[i],  # pd.Series(data=recvbuf_genes[i], index=m_index),
                       state     = None,
                       fitness   = recvbuf_fitness[i]) for i in range(config['runtime']['n_organisms'])]
    timer.end('gather_population')

     ######     ###    ##     ## ########
    ##    ##   ## ##   ##     ## ##
    ##        ##   ##  ##     ## ##
     ######  ##     ## ##     ## ######
          ## #########  ##   ##  ##
    ##    ## ##     ##   ## ##   ##
     ######  ##     ##    ###    ########

    index_best = argmax_list_of_dicts(population, 'fitness')

    comm_rank_best = index_best // n_orgsnisms_per_process
    index_best_batch = index_best % n_orgsnisms_per_process

    timer.start('save_phenotype')
    if comm_rank == comm_rank_best:

        organism_best = batch[index_best_batch]

        with open(config['runtime']['output']['organism_best_filename'], 'bw') as f:
            pickle.dump(organism_best, f)

        organism_best['genes'].to_csv(config['runtime']['output']['genes_best_filename'])

        for exp_cond_name in config['experimental_conditions']:
            if exp_cond_name == 'common':
                continue

            df = organism_best['phenotype'][exp_cond_name]

            # Rewrite last epoch
            filename_phenotype_save = os.path.join(config['runtime']['output']['output_folder_name_phenotype'],
                                                   f"phenotype_{exp_cond_name}.csv")
            df.to_csv(filename_phenotype_save, index=False)

            # Append last epoch to previous
            filename_phenotype_save_bmodelry = os.path.join(config['runtime']['output']['output_folder_name_phenotype'],
                                                          f"phenotype_{exp_cond_name}.bin")
            with open(filename_phenotype_save_bmodelry, 'ba+' if epoch else 'bw') as f:
                df.values.astype(np.float32).tofile(f)

    timer.end('save_phenotype')

    if comm_rank == epoch % comm_size:
        save_epoch(population, config['runtime']['output'])

     ######   ######## ##    ## ######## ######## ####  ######
    ##    ##  ##       ###   ## ##          ##     ##  ##    ##
    ##        ##       ####  ## ##          ##     ##  ##
    ##   #### ######   ## ## ## ######      ##     ##  ##
    ##    ##  ##       ##  #### ##          ##     ##  ##
    ##    ##  ##       ##   ### ##          ##     ##  ##    ##
     ######   ######## ##    ## ########    ##    ####  ######

    timer.start('gene')

    if len(population) <= 3:
        if comm_rank == 0:
            with open(config['runtime']['output']['log_filename'], "a") as file_log:
                file_log.write(f"# Not enough organisms for genetic operations left: {len(population)}\nexit\n")
        exit()

    population.sort(key=lambda organism: organism['fitness'], reverse=True)
    index_first_invalid = find_index_first(population, lambda organism: organism['fitness'] == np.NINF)
    if index_first_invalid:
        if comm_rank == 0:
            n_invalids = len(population) - index_first_invalid
            percentage_invalid = (n_invalids) / len(population) * 100
            with open(config['runtime']['output']['log_filename'], "a") as file_log:
                file_log.write(f"# {n_invalids} ({percentage_invalid:.2f} %) invalids were deleted\n")
        population = population[:index_first_invalid]
    elites_all = population[:config['n_elites']]  # len may be less than config['n_elites'] due to invalids
    elites_batch = elites_all[comm_rank::comm_size]
    n_elites = len(elites_batch)
    #  print(f"# {comm_rank} has {n_elites} elites", flush=True)
    #  elites_batch may be empty list

    for organism in population:
        organism['genes'], bounds_transformed = transform_genes_bounds(organism['genes'],
                                                                       bounds, gammas, mask_multipliers)
    batch = do_step(population, new_size=n_orgsnisms_per_process - n_elites,
                    elite_size=0, bounds=bounds_transformed, **config['runtime']['kw_ga'])

    batch += elites_batch

    assert (len(batch) == n_orgsnisms_per_process)

    for organism in batch:
        organism['genes'] = transform_genes_bounds_back(organism['genes'],
                                                        bounds_transformed, bounds_back=bounds,
                                                        mask_multipliers=mask_multipliers)

        organism['genes'] = pd.Series(data=organism['genes'], index=config['runtime']['m_index'])
        organism['state'] = None

    timer.end('gene')

    ########  ######## ########   #######  ########  ########
    ##     ## ##       ##     ## ##     ## ##     ##    ##
    ##     ## ##       ##     ## ##     ## ##     ##    ##
    ########  ######   ########  ##     ## ########     ##
    ##   ##   ##       ##        ##     ## ##   ##      ##
    ##    ##  ##       ##        ##     ## ##    ##     ##
    ##     ## ######## ##         #######  ##     ##    ##

    if comm_rank == epoch % comm_size:
        with open(config['runtime']['output']['log_filename'], "a") as file_log:
            file_log.write(f"# EPOCH {epoch}:\n")
            file_log.write(timer.report(sort=True) + "\n")

if comm_rank == 0:
    time_end = time.time()
    with open(config['runtime']['output']['log_filename'], "a") as file_log:
        file_log.write(f"# TIME = {time_end - time_start}")
