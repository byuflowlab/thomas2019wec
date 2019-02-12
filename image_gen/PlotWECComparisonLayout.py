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
layout_directory = '../project-code/input_files/'

layout_data = np.loadtxt(layout_directory + "layouts/round_38turbs/nTurbs38_spacing5_layout_%i.txt" % layout_number)

turbineX = layout_data[:, 0] * rotor_diameter + rotor_diameter/2.
turbineY = layout_data[:, 1] * rotor_diameter + rotor_diameter/2.

# Save the number of turbines being used.
nTurbs = turbineX.size

# Define new rotorDiameter variable to represent the rotor diameter of each turbine.
rotorDiameter = np.zeros(nTurbs)
for i in range(0, nTurbs):

    rotorDiameter[i] = rotor_diameter

# Define other necessary variables to plot the wind farm.
farm_boundary_radius = 0.5 * (rotor_diameter * 4000. / 126.4 - rotor_diameter)  # 1936.8
center = np.array([farm_boundary_radius, farm_boundary_radius]) + rotor_diameter / 2.
boundary_center_x = center[0]
boundary_center_y = center[1]

# Call PJ's function.
plot_turbine_locations(turbineX, turbineY, rotorDiameter, color='red', alpha=1.0, circle_boundary=True,
                       farm_radius=farm_boundary_radius,farm_center=(boundary_center_x, boundary_center_y),
                       boundary_color='blue')

# Plot legend and make the plot display to screen.
plt.show()
