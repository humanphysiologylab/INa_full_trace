#!/bin/bash
#SBATCH --partition normal
#SBATCH  -n40
#SBATCH --job-name pymc_task
#SBATCH --time=23:59:59
#SBATCH --chdir=.
#SBATCH --comment mcmc


cd /data90t/users/{FOLDER}/INa_full_trace/src/model_ctypes/test_pipette_M1_in/
make clean && make ina && make

cd /data90t/users/{FOLDER}/INa_full_trace/scripts/pymc_scripts
export OpenMPI
export OMP_NUM_THREADS=1
python3 run.py
