import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# indicate how many optimization runs are to be plotted
runs = 10

# Define output location
input_directory_wec = "../output_files/snopt_wec_diam_max_wec_3_nsteps_6.000/"
input_directory_snopt = "../output_files/snopt/"
input_directory_ps = "../output_files/ps/"

# Specify output file name
input_file_wec = input_directory_wec + "convergence_histories.txt"
input_file_snopt = input_directory_snopt + "convergence_histories.txt"
input_file_ps = input_directory_ps + "convergence_histories.txt"

# find how many entries are in the longest WEC convergence history
f = open(input_file_wec)
maxlength = 0
rownum = -1
for row in f.readlines():
    if rownum < 0:
        rownum += 1
        continue
    data = np.fromstring(row, sep=" ")
    if data.size > maxlength:
        maxlength = data.size
f.close()

# extract WEC convergence histories to a data frame
dfcalc_wec = pd.DataFrame({'Function Calls': np.arange(maxlength)})
dfopt_wec = pd.DataFrame({'Function Calls': np.arange(maxlength)})
rownum = -1
run = 0
f = open(input_file_wec)
for row in f.readlines():
    if rownum < 0:
        rownum += 1
        continue

    data = np.fromstring(row, sep=" ")
    if rownum % 2 == 0:
        # dfcalc.(run, run, pd.Series(data, name=rownum))
        s = pd.Series(data, name=rownum)

        dfopt_wec.insert(run, run, s)
    else:
        s = pd.Series(data, name=rownum)
        dfcalc_wec.insert(run, run, s)
        run += 1
    rownum += 1
# fill in missing values with last value
dfcalc_wec.fillna(method='ffill', inplace=True)
dfopt_wec.fillna(method='ffill', inplace=True)

# find how many entries are in the longest SNOPT convergence history
f = open(input_file_snopt)
maxlength = 0
rownum = -1
for row in f.readlines():
    if rownum < 0:
        rownum += 1
        continue
    data = np.fromstring(row, sep=" ")
    if data.size > maxlength:
        maxlength = data.size
f.close()

# extract SNOPT convergence histories to a data frame
dfcalc_snopt = pd.DataFrame({'Function Calls': np.arange(maxlength)})
dfopt_snopt = pd.DataFrame({'Function Calls': np.arange(maxlength)})
rownum = -1
run = 0
f = open(input_file_snopt)
for row in f.readlines():
    if rownum < 0:
        rownum += 1
        continue

    data = np.fromstring(row, sep=" ")
    print(data.size)
    if rownum % 2 == 0:
        # dfcalc.(run, run, pd.Series(data, name=rownum))
        s = pd.Series(data, name=rownum)
        dfopt_snopt.insert(run, run, s)
    else:
        s = pd.Series(data, name=rownum)
        dfcalc_snopt.insert(run, run, s)
        run += 1
    rownum += 1

# fill in missing values with last value
dfcalc_snopt.fillna(method='ffill', inplace=True)
dfopt_snopt.fillna(method='ffill', inplace=True)

# quit()
# find how many entries are in the longest ps convergence history
f = open(input_file_ps)
maxlength = 0
rownum = 0
maxcalls = 0
for row in f.readlines():
    if rownum < 1:
        rownum += 1
        continue

    data = np.fromstring(row, sep=" ")
    if rownum % 2 != 0:
        datamaxcalls = data[-1]
        if datamaxcalls > maxcalls:
            maxcalls = datamaxcalls
            datamax = data
    if data.size > maxlength:
        maxlength = data.size
    rownum += 1
f.close()
# extract ALPSO convergence histories to a data frame
dfcalc_ps = pd.DataFrame({'Function Calls': datamax})
# dfcalls_ps = pd.DataFrame({'Function Calls': np.arange(maxlength)})

rownum = 0
run = 0
f = open(input_file_ps)
for row in f.readlines():
    if rownum < 1:
        rownum += 1
        continue

    data = np.fromstring(row, sep=" ")
    # print(data.size)
    if rownum % 2 == 0:
        # dfcalc.(run, run, pd.Series(data, name=rownum))
        s = pd.Series(data, name=rownum)
        dfcalc_ps.insert(run, run, s)
    # else:
    #     dfcalc_ps.insert(run, run, pd.Series(data, name=rownum))
        run += 1
    rownum += 1
# print(dfcalc_ps)
# quit()
# fill in missing values with last value
dfcalc_ps.fillna(method='ffill', inplace=True)
# dfopt_ps.fillna(method='ffill', inplace=True)
print(dfcalc_snopt)
print(dfcalc_wec)
print(dfcalc_ps)
# quit()

# rearrange data frame
dfcalc_wec = dfcalc_wec.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")
dfcalc_snopt = dfcalc_snopt.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")
dfcalc_ps = dfcalc_ps.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")
dfopt_wec = dfopt_wec.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")
dfopt_snopt = dfopt_snopt.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")
# dfcalc_ps = dfcalc_ps.melt(id_vars=["Function Calls"], value_vars=np.arange(runs), var_name="run", value_name="AEP")

# concatenated_opt = pd.concat([dfopt_wec.assign(dataset='SNOPT+WEC-D'), dfopt_snopt.assign(dataset='SNOPT')])
concatenated_calc = pd.concat([dfcalc_wec.assign(dataset='SNOPT+WEC-D'), dfcalc_snopt.assign(dataset='SNOPT'), dfcalc_ps.assign(dataset="ALPSO")])

g = sns.relplot(x='Function Calls', y='AEP', kind="line", data=concatenated_calc, style='dataset', hue='dataset')
g.set(xscale="log")
plt.show()
# plt.clf()

# sns.relplot(x='index', y='AEP', kind="line", data=concatenated_opt, style='dataset', hue='dataset')
# plt.show()

# plt.plot(objectives)
# for i in np.arange(0, data.shape[0]):
#     plt.plot(data[i, :]*1E-9)
#
# plt.xlabel("Function Calls")
# plt.ylabel("AEP (GWh)")
# plt.show()