"""
Filename: JensenWECTest2.py
Author: Spencer McOmber
Description: This is an attempt at applying Jared Thomas' WEC model into Jensen's 1983 Cosine Wake Model. I have
copied and pasted the code from "Jensen3DCosineComparison.py" and will adjust the Jensen model slightly so I can
attempt to implement WEC into the model.
This file is built upon JensenWECTest1.py in that I'm trying to get the AEP vs. crosswind position rather than the
v/u vs. position. FOR SINGLE UPWIND TURBINE.
"""

from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.OptimizationGroups import AEPGroup
import numpy as np
from openmdao.api import Group, Problem
import matplotlib.pyplot as plt
from time import time

"""THIS IS THE RUN SCRIPT FOR JENSEN3D"""

tic = time()

# Create an array of the x/ro ratios used in Jensen's 1983 paper. Note that I'm only testing the ratio of x/ro = 10,
# because the previous results seemed to indicate that Jensen3D matched pretty well with all models at this ratio. No
# need to use the other ratios since I'm testing WEC and not the models' accuracy.
x_over_ro = np.array([10.0])

# Instead of looping through different x/ro ratios, this program will cycle through different values for the
# relaxation factor used in WEC. Note that the stop value is 1.0 - 0.25, or the desired stop value minus the step
# size. This is to ensure that 1.0 is included in the array.
relaxationFactor = np.arange(7.0, 0.0, -1.0)

# define turbine locations in global reference frame
# turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
# turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

# Define the start, stop, and step values for a thetaVector. Units in degrees.
thetaMax = 90
dTheta = 1.0
thetaVector = np.arange(-thetaMax, thetaMax, dTheta)


# initialize input variable arrays. Turbine coordinate arrays need to have the same sizes.
# For this run script, we only want two turbines for each run - one causing the wake, one receiving
# the wake.
nTurbines = 2

# Have the number of elements in the relaxationFactor vector as the number of rows in each turbine's position vector.
# This is so that we can create a new plot of v/u vs. crosswind position for each value of the relaxation factor we
# try. Each position vector also has thetaVector.size columns so that we can plot v/u for different theta values.
turbineX = np.zeros((relaxationFactor.size, thetaVector.size))
turbineY = np.zeros((relaxationFactor.size, thetaVector.size))
turbineYNormalized = np.zeros((relaxationFactor.size, thetaVector.size))
rotorDiameter = np.zeros(nTurbines)
axialInduction = np.zeros(nTurbines)
Ct = np.zeros(nTurbines)
Cp = np.zeros(nTurbines)
generatorEfficiency = np.zeros(nTurbines)
yaw = np.zeros(nTurbines)

# define initial values
for turbI in range(0, nTurbines):
    rotorDiameter[turbI] = 126.4            # m
    axialInduction[turbI] = 1.0/3.0
    Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
    Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    generatorEfficiency[turbI] = 0.944
    yaw[turbI] = 0.     # deg.

# Calculate the x separation distance between turbines.
rotorRadius = rotorDiameter[0] / 2.0
turbineXInitialPosition = 0.0
turbineYInitialPosition = 0.0
# Calculate the x separation distance between turbines. Calculate the y-turbine positions based on the angle theta.
for i in range(relaxationFactor.size):
    for j in range(thetaVector.size):

        # Note that I only need one other x-coordinate for this program to work; however, the OpenMDAO functions need
        # the turbine position vectors to have the same size, which is why I'm stuffing the x position vector with
        # all the same numbers.
        turbineX[i, j] = x_over_ro * rotorRadius

        # Formula for getting y-coordinates from x position and theta.
        # turbineY[i, j] = turbineYInitialPosition + turbineX[i, j] * np.arctan(np.radians(thetaVector[j]))

        # Calculate the normalized y-positions (crosswind positions).
        # turbineYNormalized[i, j] = turbineY[i, j] / rotorDiameter[0]

    # Hard-code y-coordinates for turbines.
    turbineY[i] = np.array([np.linspace(-2.0*rotorDiameter[0], 2.0*rotorDiameter[0], thetaVector.size)])
    turbineYNormalized[i] = np.array([np.linspace(-2.0, 2.0, thetaVector.size)])

# Define flow properties
nDirections = 1
wind_speed = 8.1                                # m/s
air_density = 1.1716                            # kg/m^3
# wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
wind_direction = 270.0       # deg (N = 0 deg., using direction FROM, as in met-mast data)

wind_frequency = 1.                             # probability of wind in this direction at this speed

# set up problem

# use 'variant': 'Cosine' for normal Jensen-Cosine model, use 'variant': 'CosineFortran' for PJ's FORTRAN Jensen model.
wake_model_options = {'variant': 'Cosine'}
prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                             params_IdepVar_func=add_jensen_params_IndepVarComps,
                             params_IndepVar_args={'use_angle': False}))

# initialize problem
prob.setup(check=True)

# assign values to turbine states
# prob['turbineX'] = turbineX
# prob['turbineY'] = turbineY
prob['yaw0'] = yaw

# assign values to constant inputs (not design variables)
prob['rotorDiameter'] = rotorDiameter
prob['axialInduction'] = axialInduction
prob['generatorEfficiency'] = generatorEfficiency
prob['windSpeeds'] = np.array([wind_speed])
prob['air_density'] = air_density
prob['windDirections'] = np.array([wind_direction])
prob['windFrequencies'] = np.array([wind_frequency])
prob['Ct_in'] = Ct
prob['Cp_in'] = Cp
# prob['model_params:spread_angle'] = 20.0
# prob['model_params:alpha'] = 0.1

# Save the relaxation factor defined in this run script to the OpenMDAO code.
# prob['relaxationFactor'] = relaxationFactor

# run the problem
# prob.run()

# Create a text file that I can save data into.
AEPDataFile = open('../DataFiles/JensenWECTestAEP.txt', 'w+')

# Loop through relaxation factors to calculate the v/u vs. crosswind position data for each relaxation factor.
for i in range(relaxationFactor.size):

    # For each relaxation factor, calculate the velocity deficit across all values of theta (i.e., across all values
    # of y).
    for j in range(thetaVector.size):

        # Set the appropriate x and y coordinates for the turbines.
        prob['turbineX'] = np.array([turbineXInitialPosition, turbineX[i, j]])
        prob['turbineY'] = np.array([turbineYInitialPosition, turbineY[i, j]])

        # Set the relaxation factor for this iteration.
        prob['model_params:relaxationFactor'] = relaxationFactor[i]

        # Run OpenMDAO once.
        prob.run_once()

        # Save the calculated data to a datafile.
        # velocityFile.write('%f\n' % (prob['wtVelocity0'][1] / wind_speed))
        AEPDataFile.write('%f\n' % (prob['AEP'] / 1.0e6))   # units of Giga-Watt Hours (GWh)

# Close the file for writing.
AEPDataFile.close()

toc = time()
duration = toc - tic

print 'Run-time: %f' % duration

# Reopen the velocity file I just closed so I can read it.
AEPDataFile = open('../DataFiles/JensenWECTestAEP.txt', 'r')

# Initialize a 2D numpy array that I can use to store all the v/u values from the WEC windspeed text file. Should
# have relaxationFactor.size rows and thetaVector.size columns.
AEPData = np.zeros((relaxationFactor.size, thetaVector.size))

# Initialize a Python list that will store all the label strings for the plots I'll make.
labelList = []

# Start up the figure and give it a title. I could have plotted just a figure, but the only way I could figure out to
# remove the top and right borders from the plot while keeping the figure and fonts at an appropriate size.
plt.rcParams.update({'font.size': 26})
fig = plt.figure(1, figsize=(10, 10))
ax = fig.add_subplot(111)
# plt.title('WEC Jensen Model AEP')

# Create a list of strings to use as labels based on the relaxation factors that are entered.
for i in range(relaxationFactor.size):

    # I saved the value as a float with one decimal place here.
    labelList.append(r'$\xi=%i$' % relaxationFactor[i])

# Loop through the v_over_u array and read in each line as a new value. Outer loop iterates through rows,
# inner row iterates through columns.
for i in range(relaxationFactor.size):
    for j in range(thetaVector.size):

        # Save the current line in the velocityFile into the v_over_u array.
        AEPData[i, j] = float(AEPDataFile.readline())

    # Plot the ith row's results.
    plt.plot(turbineYNormalized[i, :], AEPData[i, :], label=labelList[i])
    plt.ylabel('AEP (MWh)')
    plt.xlabel('Crosswind Position (Y/D)')

# Add a legend to the plot and display the plot.
# plt.grid(True)
plt.legend(ncol=3, frameon=False).get_frame().set_linewidth(0.0)
plt.ylim([24, 34])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# annotationLocation = (-0.5, 28.5)
# plt.annotate(r'$x/r_0=%.1f$' % x_over_ro[0], xy=annotationLocation, xytext=annotationLocation, xycoords='data')
plt.show()
