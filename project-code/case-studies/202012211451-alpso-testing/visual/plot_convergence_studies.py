import numpy as np 
import re 
from matplotlib import pyplot as plt

def parse_alpso_files(filename):
    with open(filename) as f:
        fcalls = re.findall('(?<=EVALUATIONS: ).\d+',f.read(),re.MULTILINE)
    with open(filename) as f:
        obj = re.findall('(?<=F = -).*',f.read(),re.MULTILINE)
    return obj, fcalls

if __name__ == "__main__":


    fig, ax = plt.subplots(4, 3,sharex=True)

    turbs = np.array([16,38,38,60])
    dirs = np.array([20,12,36,72])
    windrose = np.array(["directional", "nantucket", "nantucket", "amalia"])
    vinit = np.array([1,1,2])
    vcrazy = [1E-4, 1E-2, 1E-2]
    for j in np.arange(0,3):
        if j < 1:
            directory = "../output_files/ps-vinit-1p0-vcrazy-1em4/"
        else:
            directory = "../output_files/ps-vinit-%ip0/" % (vinit[j])
        for i in np.arange(0,4):
            if j<1 and i>1:
                continue
            if j<2 and i>2:
                continue
            for ii in np.arange(5,31,5):
                filename = directory+"ALPSO_summary_multistart_%iturbs_%sWindRose_%idirs_BPAModel_RunID0_TItype4_II%i_print.out" %(turbs[i],windrose[i],dirs[i],ii)
                obj, fcalls = parse_alpso_files(filename)
                obj = np.asfarray(obj,float)
                fcalls = np.asfarray(fcalls,float)
                print(obj,fcalls)
                ax[i,j].plot(fcalls, obj, label="II = %i" %(ii))
                ax[i,j].set_ylabel("Objective")
                ax[i,j].set_title("%i Turbs, %i Dirs, vinit=%i, vcrazy=%.1e" %(turbs[i],dirs[i], vinit[j], vcrazy[j]))
                ax[i,j].legend(loc=4,frameon=False)
                ax[i,j].set_xlim([0,20000])
    # fig.title("viniit = %d")
    plt.show()