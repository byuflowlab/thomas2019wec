#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=2G  # memory per CPU core
#SBATCH -J "38 turbs snopt diam wec mw4 ws 0.5"   # job name
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-199     # job array of size 100

echo ${SLURM_ARRAY_TASK_ID}
wec_method_number=1
model_number=1
op_alg_number=0
maxwec=4
wecstep=0.5

python3 opt_mstart.py ${SLURM_ARRAY_TASK_ID} $wec_method_number $model_number $op_alg_number $maxwec $wecstep
