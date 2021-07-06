#!/bin/bash
#SBATCH --job-name=21-avo
#SBATCH --partition mix
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=11:59:59
#SBATCH --chdir=.

cd /data4t/student/21-AVO/INa_full_trace/ga/pypoptim
echo A
python mpi_script.py configs/config_test.json
echo B
