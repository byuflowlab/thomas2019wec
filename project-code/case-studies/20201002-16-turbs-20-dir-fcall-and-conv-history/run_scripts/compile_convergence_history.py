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

get_conv = False
get_steps = True 

# Define output location
output_directory_wec = "../output_files/snopt_wec_diam_max_wec_3_nsteps_6.000/"
output_directory_snopt = "../output_files/snopt/"
output_directory_ps = "../output_files/ps/"
if get_conv:
    # Specify conv output file name 
    output_file_wec = output_directory_wec + "convergence_histories.txt"
    output_file_snopt = output_directory_snopt + "convergence_histories.txt"
    output_file_ps = output_directory_ps + "convergence_histories.txt"
if get_steps:
    # Specify conv output file name 
    step_output_file_wec = output_directory_wec + "step_breakdown.txt"
    step_output_file_snopt = output_directory_snopt + "step_breakdown.txt"
    step_output_file_ps = output_directory_ps + "step_breakdown.txt"

# indicate how many optimization runs are to be compiled
runs = 10

# compile SNOPT+WEC convergence history
for run_number in np.arange(0, runs):

    data_in = np.loadtxt(output_directory_wec+"convergence_history_run%i.txt" %(run_number))

    nits = data_in.shape[0]
    AEPopt = data_in[:, 0]
    AEPcalc = data_in[:, 1]

    f = open(output_file_wec, "a")
    if run_number == 0:
        header = "convergence history alternating row AEP (W) unadjusted, AEP (W) adjusted"
    else:
        header = ""
    np.savetxt(f, (AEPopt, AEPcalc), header=header)
    f.close()

# compile SNOPT convergence history
for run_number in np.arange(0, runs):

    data_in = np.loadtxt(output_directory_snopt+"convergence_history_run%i.txt" %(run_number))

    nits = data_in.shape[0]
    AEPopt = data_in[:, 0]
    AEPcalc = data_in[:, 1]

    f = open(output_file_snopt, "a")
    if run_number == 0:
        header = "convergence history alternating row AEP (W) unadjusted, AEP (W) adjusted"
    else:
        header = ""
    np.savetxt(f, (AEPopt, AEPcalc), header=header)
    f.close()

# compile ALPSO convergence history
for run_number in np.arange(0, runs):

    fcalls, obj = get_alpso_history(output_directory_ps + "ALPSO_summary_multistart_16turbs_directionalWindRose_20dirs_BPAModel_RunID%i_print.out" %(run_number))

    nits = len(obj)
    AEPcalc = obj*1E7

    f = open(output_file_ps, "a")
    if run_number == 0:
        header = "convergence history alternating row function calls, AEP (W)"
    else:
        header = ""
    np.savetxt(f, (fcalls, -AEPcalc), header=header)
    f.close()




# if __name__ == "__main__":
#     filepath = "../output_files/ps/ALPSO_summary_multistart_16turbs_directionalWindRose_20dirs_BPAModel_RunID1_print.out"
#     fcalls, obj = get_alpso_history(filepath)
#     print(fcalls, obj)
#     plt.plot(fcalls, obj)
#     plt.show()