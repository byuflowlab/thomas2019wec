"""
Filename: BastankhahWECTest1.py
Author: Spencer McOmber
Created: Nov. 13, 2018
Description: This file is meant to be a run script for the Bastankhah & Porte-Agel turbine wake model. The purpose is to experiment applying Jared Thomas'
WEC idea to the FLORISSE wake model. It is hoped that applying WEC will allow us to spread the wake of each turbine,
which aids in gradient-based optimization of each turbine's position to maximize the wind farm's AEP.
This run script's specific purpose is to obtain a v/u curve (or AEP curve) vs. crosswind position for a SINGLE turbine.
THIS RUN SCRIPT WAS COPIED AND PASTED FROM "FLORISSE_WECTest1.py" - NEEDS TO BE ADAPTED TO Bastankhah.

Run script obtained from "test_gradients.py" from the "TotalDerivTestsFlorisAEPOptRotor" class. This file is found
under "tests" directory under "FLORISSE".
"""

from __future__ import print_function
import unittest
from openmdao.api import pyOptSparseDriver, Problem, ScipyOptimizer
from plantenergy.OptimizationGroups import *
from plantenergy.GeneralWindFarmComponents import calculate_boundary
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
import matplotlib.pyplot as plt

import cPickle as pickle

from scipy.interpolate import UnivariateSpline

nTurbines = 2
# nTurbines = 4
# self.rtol = 1E-6
# self.atol = 1E-6

# Add in necessary variables. NOT SURE HOW TO ADD IN RELAXATION FACTOR FOR FLORISSE YET, SO I'LL JUST MAKE IT EQUAL
# TO ONE FOR NOW.
x_over_ro = np.array([10.0])
relaxationFactor = np.arange(7.0, 0.0, -1.0)
# relaxationFactor = np.array([3.0, 1.0])

# Define the start, stop, and step values for a thetaVector. Units in degrees.
thetaMax = 90
dTheta = 1.0
thetaVector = np.arange(-thetaMax, thetaMax, dTheta)

# np.random.seed(seed=10)

# turbineX = np.random.rand(nTurbines)*3000.
# turbineY = np.random.rand(nTurbines)*3000.

minSpacing = 2

# Have the number of elements in the relaxationFactor vector as the number of rows in each turbine's position vector.
# This is so that we can create a new plot of v/u vs. crosswind position for each value of the relaxation factor we
# try. Each position vector also has thetaVector.size columns so that we can plot v/u for different theta values.
turbineX = np.zeros((relaxationFactor.size, thetaVector.size))
turbineY = np.zeros((relaxationFactor.size, thetaVector.size))
turbineYNormalized = np.zeros((relaxationFactor.size, thetaVector.size))
rotorDiameter = np.zeros(nTurbines)
axialInduction = np.zeros(nTurbines)
generatorEfficiency = np.zeros(nTurbines)
yaw = np.zeros(nTurbines)
hubHeight = np.zeros(nTurbines)
Ct = np.zeros(nTurbines)
Cp = np.zeros(nTurbines)

# initialize input variable arrays
# rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
# axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
# generatorEfficiency = np.ones(nTurbines)*np.random.random()
# yaw = np.random.rand(nTurbines)*60. - 30.
for turbI in range(0, nTurbines):
    rotorDiameter[turbI] = 126.4            # m
    axialInduction[turbI] = 1.0/3.0
    generatorEfficiency[turbI] = 0.944
    yaw[turbI] = 0.     # deg.
    hubHeight[turbI] = 90.     # deg.
    Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
    Cp[turbI] = 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)

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
        turbineX[i, j] = x_over_ro[0] * rotorRadius

    # Hard-code y-coordinates for turbines.
    turbineY[i] = np.array([np.linspace(-2.0*rotorDiameter[0], 2.0*rotorDiameter[0], thetaVector.size)])
    turbineYNormalized[i] = np.array([np.linspace(-2.0, 2.0, thetaVector.size)])

# Define flow properties
# nDirections = 50
# windSpeeds = np.random.rand(nDirections)*20        # m/s
# air_density = 1.1716    # kg/m^3
# windDirections = np.random.rand(nDirections)*360.0
# windFrequencies = np.random.rand(nDirections)
nDirections = 1
wind_speed = np.array([8.1])
air_density = 1.1716    # kg/m^3
wind_direction = np.array([270.0])
wind_frequency = np.array([1.0])

# Final variable is for the cosine spread that Jared Thomas added in with his paper from 2017.
w = 2.0

# set up problem
# prob = Problem(root=OptAEP(nTurbines, nDirections=1))

# prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=windDirections.size,
#                            minSpacing=minSpacing, use_rotor_components=True))
# prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=wind_direction.size,
#                            minSpacing=minSpacing, use_rotor_components=True))
wake_model_options = {'nSamples': 0}
prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=gauss_wrapper,
                                     wake_model_options=wake_model_options, datasize=0, use_rotor_components=False,
                                     params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                     params_IndepVar_args={}))

# initialize problem
prob.setup(check=True)

# assign values to constant inputs (not design variables)
NREL5MWCPCT = pickle.load(open('../input_files/NREL5MWCPCT_smooth_dict.p'))
# prob['model_params:cos_spread'] = w # added this line of code based on what Jared T. told me on Nov. 7. See his 2017
# paper for more info.
prob['model_params:z_ref'] = 90.    # found this in a unit test within gaussian-wake, so I thought I'd include it.

prob['yaw0'] = yaw
prob['hubHeight'] = hubHeight
prob['rotorDiameter'] = rotorDiameter
prob['axialInduction'] = axialInduction
prob['generatorEfficiency'] = generatorEfficiency
prob['windSpeeds'] = wind_speed
prob['air_density'] = air_density
prob['windDirections'] = wind_direction
prob['windFrequencies'] = wind_frequency
# prob['model_params:FLORISoriginal'] = False
prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
prob['Ct_in'] = Ct
prob['Cp_in'] = Cp

# prob['Ct_in'] = np.ones(nTurbines) * 0.5
# prob['Cp_in'] = np.ones(nTurbines) * 0.4

# assign values to turbine states
# prob['turbineX'] = turbineX
# prob['turbineY'] = turbineY
# prob['hubHeight'] = hubHeight
# prob['yaw0'] = yaw
#
# # assign values to constant inputs (not design variables)
# prob['rotorDiameter'] = rotorDiameter
# prob['axialInduction'] = axialInduction
# prob['generatorEfficiency'] = generatorEfficiency
# prob['windSpeeds'] = np.array([wind_speed])
# prob['model_params:z_ref'] = 90.
# prob['air_density'] = air_density
# prob['windDirections'] = np.array([wind_direction])
# prob['windFrequencies'] = np.array([wind_frequency])
# prob['Ct_in'] = Ct
# prob['Cp_in'] = Cp

# prob.setup(check=True)

# Create a text file that I can save data into.
VelocityDataFile = open('../DataFiles/BastankhahWECTestVelocity.txt', 'w+')

# Loop through relaxation factors to calculate v/u vs. crosswind position.
for i in range(relaxationFactor.size):

    # For each relaxation factor, calculate the velocity deficit across all values of Y.
    for j in range(thetaVector.size):

        prob['turbineX'] = np.array([turbineXInitialPosition, turbineX[i, j]])
        prob['turbineY'] = np.array([turbineYInitialPosition, turbineY[i, j]])

        # print(prob['turbineX'], prob['turbineY'])
        # print(np.array([turbineXInitialPosition, turbineX[i, j]]), np.array([turbineYInitialPosition, turbineY[i, j]]))

        # Set the relaxation factor for this iteration.
        prob['model_params:opt_exp_fac'] = relaxationFactor[i]

        # Run OpenMDAO once.
        prob.run_once()
        # prob.run()

        # Save the calculated data to a datafile.
        VelocityDataFile.write('%f\n' % (prob['wtVelocity0'][-1] / wind_speed[0]))

        # print('wind turbine velocity', prob['wtVelocity0'])

# Close the file for writing.
VelocityDataFile.close()

# Reopen the velocity file so I can read it.
VelocityDataFile = open('../DataFiles/BastankhahWECTestVelocity.txt', 'r')

# Initialize a 2D numpy array that I can use to store all the v/u values from the WEC windspeed text file. Should
# have relaxationFactor.size rows and thetaVector.size columns.
VelocityData = np.zeros((relaxationFactor.size, thetaVector.size))

# Initialize a Python list that will store all the label strings for the plots I'll make.
labelList = []

# Start up the figure and give it a title.
plt.figure(1, figsize=(9, 9))
plt.title('WEC Bastankhah Model V/U')

# Create a list of strings to use as labels based on the relaxation factors that are entered.
for i in range(relaxationFactor.size):

    # I saved the value as a float with one decimal place here.
    labelList.append(r'$\xi=%.2f$' % relaxationFactor[i])

# Loop through the v_over_u array and read in each line as a new value. Outer loop iterates through rows,
# inner row iterates through columns.
for i in range(relaxationFactor.size):
    for j in range(thetaVector.size):

        # Save the current line in the velocityFile into the v_over_u array.
        VelocityData[i, j] = float(VelocityDataFile.readline())

    # Plot the ith row's results.
    plt.plot(turbineYNormalized[i, :], VelocityData[i, :], label=labelList[i])
    plt.ylabel('Velocity Ratio (V/U)')
    plt.xlabel('Crosswind Position (Y/D)')

# Add a legend to the plot and display the plot.
plt.grid(True)
plt.legend(ncol=2)
annotationLocation = (-0.5, 0.92)   # change these coordinates once the plot is fixed.
plt.annotate(r'$x/r_0=%.1f$' % x_over_ro[0], xy=annotationLocation, xytext=annotationLocation, xycoords='data')
plt.show()

# pass results to self for use with unit test
# self.J = prob.check_total_derivatives(out_stream=None)
# self.nDirections = nDirections

# print(self.J)
