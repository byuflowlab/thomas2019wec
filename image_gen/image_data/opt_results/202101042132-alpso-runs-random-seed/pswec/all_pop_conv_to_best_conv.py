import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nturbs = 38
ndirs = 12
aept = 4994091.77684705*nturbs*1E3 #Wh
psscale = -1E5
wakemodel = "BPA"
alpha = 0.08
markeralpha = 0.5
colors = ['b', 'c', 'r','m']

# initalize plot
fig1, ax1 = plt.subplots(1)

# find how many entries are in the longest WEC convergence history
input_file_alpso_wec_all = "convergence_histories_all_pop.txt"
f = open(input_file_alpso_wec_all)
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
rownum = -1
run = 0
f = open(input_file_alpso_wec_all)
for row in f.readlines():
    if rownum < 0:
        rownum += 1
        continue

    data = np.fromstring(row, sep=" ")
    if rownum % 2 == 0:
        s = pd.Series(data, name=rownum)
    else:
        s = pd.Series(data, name=rownum)
        ax1.semilogx(np.arange(1, s.size+1), 100*(1-s/aept), alpha=alpha, color=colors[0], zorder=1)
        # ax1.plot(np.arange(1, s.size+1), 100*(1-s/aept), alpha=alpha, color=colors[0], zorder=1)
        ax1.scatter(s.size, 100*(1-s.iloc[-1]/aept), marker='o', edgecolor='k', color=colors[0], zorder=10, alpha=markeralpha)
        
        run += 1
    rownum += 1
f.close()

plt.show()