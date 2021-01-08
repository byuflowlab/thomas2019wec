import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

def get_alpso_history(filepath):

    fcall_pattern = re.compile("(?<=NUMBER OF OBJECTIVE FUNCTION EVALUATIONS: ).*")
    obj_pattern = re.compile("(?<=	F = ).*")

    nresults = 0
    for i, line in enumerate(open(filepath)):
        for match in re.finditer(fcall_pattern, line):
            nresults += 1

    # print("nresults ", nresults)
    fcalls = np.zeros(nresults)
    obj = np.zeros(nresults)
    count = 0
    for j, line in enumerate(open(filepath)):

        for match in re.finditer(fcall_pattern, line):
            # print("fcalls ", match.group())
            fcalls[count] = match.group()
        for match in re.finditer(obj_pattern, line):
            # print("obj ", match.group())
            obj[count] = np.copy(match.group())
            count += 1

    # print(fcalls, obj)
    return fcalls, obj

# Define output location
output_directory_ps = "../output_files/ps/"
turbs = np.array([16,38,38,60])
dirs = np.array([20,12,36,72])
windrose = np.array(["directional", "nantucket", "nantucket", "amalia"])
scale = np.array([1E11,1E12,1E12,1E12])

for i in np.arrange(0,4):
        
    # Specify output file name
    output_file_ps = output_directory_ps + "convergence_histories_%iturbs_%idirs.txt" %(turbs[i],dirs[i])

    # indicate how many optimization runs are to be compiled
    runs = 200

    # compile ALPSO convergence history
    for run_number in np.arange(0, runs):

        fcalls, obj = get_alpso_history(output_directory_ps + "ALPSO_summary_multistart_%iturbs_%sWindRose_%idirs_BPAModel_RunID%i_print.out" %(turbs[i], windrose[i], dirs[i], run_number))

        nits = len(obj)
        AEPcalc = obj*1E7

        f = open(output_file_ps, "a")
        if run_number == 0:
            header = "convergence history alternating row function calls, AEP (W)"
        else:
            header = ""
        np.savetxt(f, (fcalls, -AEPcalc), header=header)
        f.close()