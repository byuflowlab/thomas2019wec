#!/bin/bash

#SBATCH --time=00:05:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "JensenTest"   # job name
#SBATCH --qos=test
#SBATCH --array=0     # job array of size 1

echo ${SLURM_ARRAY_TASK_ID}

mpirun python opt_snopt_relax.py ${SLURM_ARRAY_TASK_ID}