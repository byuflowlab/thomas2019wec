#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=1G  # memory per CPU core
#SBATCH -J "16 turbs snopt diam wec"   # job name
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-199     # job array of size 100
#SBATCH -C rhel7

echo ${SLURM_ARRAY_TASK_ID}
wec_method_number=1
model_number=1
op_alg_number=0

mpirun python3 opt_mstart.py ${SLURM_ARRAY_TASK_ID} $wec_method_number $model_number $op_alg_number
