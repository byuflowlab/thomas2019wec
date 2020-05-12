import numpy as np

def write_scripts(filename, hours=24, tasks=1, mem="1G", email='jaredthomas68@gmail.com', maxwec=1, nsteps=1, wec_method='none', name="", opt_alg='snopt', wake_model='BPA', path='./'):

    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    wake_model_num = MODELS.index(wake_model)

    opt_algs = ['snopt', 'ga', 'ps']
    opt_alg_num = opt_algs.index(opt_alg)

    wec_methods = ['none', 'diam', 'angle', 'hybrid']
    wec_method_num = wec_methods.index(wec_method)

    f = open(path+filename, "w")

    # add shebang and white space
    f.write("#!/bin/bash\n")
    f.write("\n")

    # add slurm commands
    f.write("#SBATCH --time=%i:00:00   # walltime\n" %(int(hours)))
    f.write("#SBATCH --ntasks=%i   # number of processor cores per sub-job(i.e. tasks)\n" %(int(tasks)))
    f.write("#SBATCH --mem-per-cpu=%s  # memory per CPU core\n"  %(mem))
    name = "38 turbs %s. alg: %s. wec method: %s. ns:%i. mw:%i." %(name, opt_alg, wec_method, nsteps, maxwec)
    f.write("#SBATCH -J '%s # job name'\n" % (name))
    if email != None:
        f.write("#SBATCH --mail-user=%s   # email address\n" %(email))
        f.write("#SBATCH --mail-type=BEGIN\n")
        f.write("#SBATCH --mail-type=END\n")
    f.write("#SBATCH --array=0-199     # job array of size 200\n\n")

    # state which member of the array is running
    f.write("echo ${SLURM_ARRAY_TASK_ID}\n\n")

    # assign inputs to opt_mstart.py
    f.write("model_number=%i\n" %(wake_model_num))
    f.write("op_alg_number=%i\n" %(opt_alg_num))

    f.write("wec_method_number=%i\n" %(wec_method_num))
    f.write("maxwec=%i\n" %(maxwec))
    f.write("nsteps=%i\n\n" %(nsteps))

    f.write("python3 opt_mstart.py ${SLURM_ARRAY_TASK_ID} $wec_method_number $model_number $op_alg_number $maxwec $nsteps\n")

    f.close()

    return

if __name__ == "__main__":

    name = "max wec, constant nsteps"

    maxweca = 3
    maxwecd = 3
    maxwech = 3

    # wecamaxvals = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    wecasteps = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    wecdsteps = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    wechsteps = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

    for i in np.arange(0, wecasteps.size):
        wec_method = 'angle'
        filename = "run_"+wec_method+"_wec_mw%i_ns%i.sh" %(maxweca, wecasteps[i])
        print("writing file: ", filename)
        write_scripts(filename, name=name, wec_method=wec_method, maxwec=maxweca, nsteps=wecasteps[i])

    for i in np.arange(0, wecdsteps.size):
        wec_method = 'diam'
        filename = "run_"+wec_method+"_wec_mw%i_ns%i.sh" %(maxwecd, wecdsteps[i])
        print("writing file: ", filename)
        write_scripts(filename, name=name, wec_method=wec_method, maxwec=maxwecd, nsteps=wecdsteps[i])

    for i in np.arange(0, wechsteps.size):
        wec_method = 'hybrid'
        filename = "run_"+wec_method+"_wec_mw%i_ns%i.sh" %(maxwech, wechsteps[i])
        print("writing file: ", filename)
        write_scripts(filename, name=name, wec_method=wec_method, maxwec=maxwech, nsteps=wechsteps[i])

    print("complete")