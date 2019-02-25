"""
Filename:       PlotWECComparisonLayout.py
Created:        Feb 11, 2019
Author:         Spencer McOmber
Description:    Use PJ and Eduardo's wind-farm-utilities repo and visualization tools to plot the initial and final
layouts from the WEC optimization comparison study.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import PJ Stanley's code for visualizing wind farm layouts.
from wind_farm_visualization import plot_turbine_locations

# Initialize variables.
turbine_type = 'NREL5MW'            #can be 'V80' or 'NREL5MW'
layout_number = 0

# Assign variables based on turbine_type.
if turbine_type == 'V80':

    rotor_diameter = 80.  # (m)

elif turbine_type == 'NREL5MW':

    rotor_diameter = 126.4  # (m)

# load starting locations
# layout_directory = '../project-code/input_files/'
#
# layout_data = np.loadtxt(layout_directory + "layouts/round_38turbs/nTurbs38_spacing5_layout_%i.txt" % layout_number)
#
# turbineX = layout_data[:, 0] * rotor_diameter + rotor_diameter / 2.
# turbineY = layout_data[:, 1] * rotor_diameter + rotor_diameter / 2.
#
# print('turbineX', turbineX)
# print('turbineY', turbineY)

# Desired model to draw layout data from.
MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
FLORIS = 0
BPA = 1
JENSEN = 2
LARSEN = 3
model = JENSEN
modelString = MODELS[model] + '_wec_opt/'

# Whether or not to use the WEC data or the non-WEC data.
relax = True
if relax:
    wecString = '_wec'
else:
    wecString = ''

# Add a string to indicate which optimization attempt (i.e., try) that we want to plot from. The "_try2" folder has
# the good Jensen optimization results.
# whichOptimizationTry = '_try2/'
whichOptimizationTry = '/'

# Specify the path to the layout directory.
layout_directory = '../project-code/optimizations/' + modelString + 'output_files_snopt' + wecString + \
                   whichOptimizationTry

# Load the data from the appropriate file in this directory.
layout_data = np.loadtxt(layout_directory + 'snopt_multistart_locations_38turbs_nantucketWindRose_12dirs_JENSEN_run' +
                                            '%i_EF1.000_TItype0.txt' % layout_number)

turbineXInit = layout_data[:, 0]
turbineYInit = layout_data[:, 1]
turbineXFinal = layout_data[:, 2]
turbineYFinal = layout_data[:, 3]

print('turbineXInit', turbineXInit)
print('turbineYInit', turbineYInit)
print('turbineXFinal', turbineXFinal)
print('turbineYFinal', turbineYFinal)

# Save the number of turbines being used.
# nTurbs = turbineX.size
nTurbs = turbineXInit.size

# Define new rotorDiameter variable to represent the rotor diameter of each turbine.
rotorDiameter = np.zeros(nTurbs)
for i in range(0, nTurbs):

    rotorDiameter[i] = rotor_diameter

# Define other necessary variables to plot the wind farm.
farm_boundary_radius = 0.5 * (rotor_diameter * 4000. / 126.4 - rotor_diameter)  # 1936.8
center = np.array([farm_boundary_radius, farm_boundary_radius]) + rotor_diameter / 2.
boundary_center_x = center[0]
boundary_center_y = center[1]

# Call PJ's function to plot the INITIAL coordinates.
# plot_turbine_locations(turbineX, turbineY, rotorDiameter, color='red', alpha=1.0, circle_boundary=True,
#                        farm_radius=farm_boundary_radius, farm_center=(boundary_center_x, boundary_center_y),
#                        boundary_color='blue')
plot_turbine_locations(turbineXInit, turbineYInit, rotorDiameter, color='red', alpha=1.0, circle_boundary=True,
                       farm_radius=farm_boundary_radius, farm_center=(boundary_center_x, boundary_center_y),
                       boundary_color='blue')

# Plot legend and make the plot display to screen.
plt.show()

# Wait for user to tell program to continue.
# plt.waitforbuttonpress()

# Now clear the figure and replot using FINAL coordinates.
plt.clf()
plot_turbine_locations(turbineXFinal, turbineYFinal, rotorDiameter, color='red', alpha=1.0, circle_boundary=True,
                       farm_radius=farm_boundary_radius, farm_center=(boundary_center_x, boundary_center_y),
                       boundary_color='blue')

# Display the plot.
plt.show()
