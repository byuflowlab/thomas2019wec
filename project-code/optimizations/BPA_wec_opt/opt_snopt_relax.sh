#!/bin/bash

#SBATCH --time=15:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=500G   # memory per CPU core
#SBATCH -J "38 turbs BPA snopt relax"   # job name
#SBATCH --mail-user=spencer.mcomber@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-199     # job array of size 100

echo ${SLURM_ARRAY_TASK_ID}

mpirun python opt_snopt_relax.py ${SLURM_ARRAY_TASK_ID}