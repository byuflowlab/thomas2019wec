import numpy as np

if __name__ == "__main__":


    opt_alg = 'snopt'
    type = 'multistart'
    # type = 'relax'
    # relax = True
    relax = False
    nruns = 200
    nturbs = 38
    windrose = 'nantucket'
    ndirs = 12
    model = 'FLORIS'

    outfile = '%s_multistart_rundata_%iturbs_%sWindRose_%idirs_%s_all.txt' % (opt_alg, nturbs, windrose, ndirs, model)

    if relax:
        header = "# run number, exp fac, ti calc, ti opt, aep init calc (kW), aep init opt (kW), aep run calc (kW), " \
                 "aep run opt (kW), run time (s), obj func calls, sens func calls"
    else:
        header = "# run number, ti calc, ti opt, aep init calc (kW), aep init opt (kW), aep run calc (kW), " \
                 "aep run opt (kW), run time (s), obj func calls, sens func calls"

    f = open(outfile, 'a')
    for i in np.arange(0, nruns):
        filename = '%s_multistart_rundata_%iturbs_%sWindRose_%idirs_%s_run%i.txt' % (
            opt_alg, nturbs, windrose, ndirs, model, i)
        try:
            data = np.loadtxt(filename)
	    # print data.shape
        except:
            print "failed to open file: "+filename
            continue
        if i == 0:
            if relax:
                np.savetxt(f, np.c_[data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7],
                                    data[:, 8],data[:,9],data[:,10]], header=header)
            else:
                np.savetxt(f, np.c_[[data]], header=header)
        else:
            if relax:
                np.savetxt(f, np.c_[data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7],
                                    data[:, 8],data[:,9],data[:,10]], header='')
            else:
                np.savetxt(f, np.c_[[data]], header='')

    f.close()

    print "finished reading and saving data"

