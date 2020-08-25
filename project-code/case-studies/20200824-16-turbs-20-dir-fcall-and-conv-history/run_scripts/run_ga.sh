#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=1G  # memory per CPU core
#SBATCH -J '16 turbs. alg: ga. wec method: none.# job name'
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --array=0-199     # job array of size 200

echo ${SLURM_ARRAY_TASK_ID}

model_number=1
op_alg_number=1
wec_method_number=0
maxwec=1
nsteps=1

python3 opt_mstart.py ${SLURM_ARRAY_TASK_ID} $wec_method_number $model_number $op_alg_number $maxwec $nsteps
