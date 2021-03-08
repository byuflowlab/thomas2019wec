#table generation practice
import numpy as np
import pandas as pd

n = 3
# create synthetic data
a = np.zeros(n, dtype=int)
b = np.ones(n)
c = np.ones(n)*2.1567890
d = np.ones(n)*3
e = np.ones(n)*4
f = np.ones(n)*5
g = np.ones(n)*6
h = np.empty(n, dtype=object)
h[:] = np.nan
h[1] = "$< %.3f$" %(0.001)
print(h)
# put data in array for adding to dataframe
data = np.array([a, b, c, d, e, f, g, h])

# row names
rows = ["A", "B", "C"]

# column names
cols0 = ["AA", "AA", "AA", "BB", "BB", "BB", "BB", "BB"]
cols1 = ["a", "b", "c", "d", "e", "f", "g", "h"]
def formatter(element):
    if type(element) is float:
        return "%.3f" % element
    elif type(element) is int:
        return "%i" % element
    else:
        return element

formatters = [formatter, formatter, formatter, formatter, formatter, formatter, formatter, formatter]
# combine column names into tuples for handing to pandas Multiindex
colstuples = list(zip(cols0,cols1))

# create multi-level column names
colsindex = pd.MultiIndex.from_tuples(colstuples, names=["", ""])

# generate pandas dataframe
df = pd.DataFrame(data.T, index=rows, columns=colsindex)

# print data frame to check it is correct
print(df)

# generate latex table code of data frame
print(df.to_latex(caption="Test Table", na_rep="", column_format="lrrrrrrrr", formatters=formatters, float_format="{:0.3f}".format, multicolumn_format='c'))