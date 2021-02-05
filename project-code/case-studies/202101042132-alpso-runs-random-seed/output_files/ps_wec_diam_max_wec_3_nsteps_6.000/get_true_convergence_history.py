import numpy as np 
import regex as re

def parse_alpso_file(path, nturbines):
 
    # parse file for objective values to determine how many steps in the run
    with open(path) as f:
        obj = re.findall('(?<=F = -).*',f.read(),re.MULTILINE)

    # find how many steps in optimization
    steps = len(obj)
    print(steps)

    # initialize position array with correct shape steps by turbines
    xpositions = np.zeros([steps, nturbines])
    ypositions = np.zeros([steps, nturbines])

    # parse file for all turbine locations at each step
    for i in np.arange(0, nturbines):
        with open(path) as f:
            # find the location fot he given wind turbine at all steps
            pattern = re.compile('(?<=P\\(%i\\) = )\d+.\d+' %(i))
            p = re.findall(pattern, f.read())

        # save positions for this turbine at all steps
        xpositions[:, i] = p

    for i in np.arange(nturbines, 2*nturbines):
        with open(path) as f:
            # find the location fot he given wind turbine at all steps
            pattern = re.compile('(?<=P\\(%i\\) = )\d+.\d+' %(i))
            p = re.findall(pattern, f.read())

        # save positions for this turbine at all steps
        ypositions[:, i-nturbines] = p
    
    return xpositions, ypositions

def get_position_set(directory, runid, expansion_factors=np.append(np.flip(np.linspace(1,3,6)),1), titypes=None):
    if titypes == None:
        titypes = np.zeros_like(expansion_factors)
        titypes[-1] = 5

def calculate_aep_set(positions):
    return
def get_convergencec_histories(directory):
    return

if __name__ == "__main__":

    directory = "./"
    filename = "ALPSO_summary_multistart_38turbs_nantucketWindRose_12dirs_BPAModel_RunID0_EF3.000_TItype0_print.out"
    path = directory+filename
    xpositions, ypositions = parse_alpso_file(path, 38)
    print(xpositions)
    print(ypositions)


