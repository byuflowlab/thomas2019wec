"""
Filename: JensenWECTest3.py
Author: Spencer McOmber
Description: This is an attempt at applying Jared Thomas' WEC model into Jensen's 1983 Cosine Wake Model. I have
copied and pasted the code from "Jensen3DCosineComparison.py" and will adjust the Jensen model slightly so I can
attempt to implement WEC into the model.
This file is built upon JensenWECTest2.py in the hopes of extracting a reasonable optimization for the turbines' x
and y positions. Only two turbines will be optimized in this file to make sure that the optimization is working
properly. Because I'm testing multiple relaxation factors, I'm expecting the same number of optimized coordinate sets as relaxation factors.
"""

from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.OptimizationGroups import AEPGroup, OptAEP
import numpy as np
from openmdao.api import Group, Problem, ScipyOptimizer, pyOptSparseDriver
from pyoptsparse import Optimization, OPT, SNOPT
import matplotlib.pyplot as plt
import time

"""THIS IS THE RUN SCRIPT FOR JENSEN3D"""

# Create an array of the x/ro ratios used in Jensen's 1983 paper. Note that I'm only testing the ratio of x/ro = 10,
# because the previous results seemed to indicate that Jensen3D matched pretty well with all models at this ratio. No
# need to use the other ratios since I'm testing WEC and not the models' accuracy.
x_over_ro = np.array([10.0])

# Instead of looping through different x/ro ratios, this program will cycle through different values for the
# relaxation factor used in WEC. Note that the stop value is 1.0 - 0.25, or the desired stop value minus the step
# size. This is to ensure that 1.0 is included in the array.
relaxationFactor = np.arange(3.0, 0.75, -0.5)

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
# turbineZ = np.zeros((relaxationFactor.size, thetaVector.size))
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
    Cp[turbI] = 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
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

        # Enter the height (z) coordinate of each turbine's hub.
        # turbineZ[i, j] = 150.0

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

wake_model_options = {'variant': 'Cosine'}
prob = Problem(root=OptAEP(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                             params_IdepVar_func=add_jensen_params_IndepVarComps,
                             params_IndepVar_args={'use_angle': False}))

# Set up the normal driver.
prob.driver = ScipyOptimizer()
prob.driver.options['optimizer'] = 'SLSQP'

# Set up SNOPT driver.
# prob.driver = pyOptSparseDriver()
# prob.driver.options['optimizer'] = 'SNOPT'

# Add design variables.
prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*0, upper=np.ones(nTurbines)*1000)
prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*(-750), upper=np.ones(nTurbines)*750)
# prob.driver.add_desvar('z', lower=np.ones(nTurbines)*100, upper = np.ones(nTurbines)*200)
prob.driver.add_objective('obj')    # need to call a flag. don't call 'AEP' directly.

# # initialize problem
prob.setup()

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
# prob['Ct_in'] = Ct
# prob['Cp_in'] = Cp
# prob['model_params:spread_angle'] = 20.0
# prob['model_params:alpha'] = 0.1

# Save the relaxation factor defined in this run script to the OpenMDAO code.
# prob['relaxationFactor'] = relaxationFactor

# run the problem
# prob.run()

# Create a text file that I can save data into.
# AEPDataFile = open('../DataFiles/JensenWECTestAEP.txt', 'w+')

# SET UP THE PROBLEM MORE IN ORDER TO OPTIMIZE.
# Run one iteration of the optimization to have a starting point?
secondTurbineXInitialPosition = 100.0
secondTurbineYInitialPosition = 10.0

# Set the appropriate x and y coordinates for the turbines.
# prob['turbineX'] = np.array([turbineXInitialPosition, turbineX[0, 0]])
# prob['turbineY'] = np.array([turbineYInitialPosition, turbineY[0, 0]])
prob['turbineX'] = np.array([turbineXInitialPosition, secondTurbineXInitialPosition])
prob['turbineY'] = np.array([turbineYInitialPosition, secondTurbineYInitialPosition])

print 'Initial turbine x-coordinates:', prob['turbineX']
print 'Initial turbine y-coordinates:', prob['turbineY']

# Assign values to parameters. Let's just choose relaxationFactor = 1.0 for this optimization.
# prob['x'] = np.array([turbineXInitialPosition, secondTurbineXInitialPosition])
# prob['y'] = np.array([turbineYInitialPosition, secondTurbineYInitialPosition])
# prob['z'] = np.array([150.0, 150.0])
# prob['r'] = rotorRadius
prob['model_params:relaxationFactor'] = relaxationFactor[-1]

# Set up the normal driver.
# prob.driver = ScipyOptimizer()
# prob.driver.options['optimizer'] = 'SLSQP'

# # Set up SNOPT driver.
# prob.driver = pyOptSparseDriver()
# prob.driver.options['optimizer'] = 'SNOPT'
#
# # Add design variables.
# prob.driver.add_desvar('turbineX', lower=np.ones(nTurbines)*0, upper=np.ones(nTurbines)*1000)
# prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*(-750), upper=np.ones(nTurbines)*750)
# # prob.driver.add_desvar('z', lower=np.ones(nTurbines)*100, upper = np.ones(nTurbines)*200)
# prob.driver.add_objective('AEP')

# Looks like I need to call "prob.setup()" again, based on JensenOpenMDAO.py.
# prob.setup()

# initialize problem
prob.setup(check=True)

# Run the problem.
print 'start Jensen run'
tic = time.time()
prob.run()
toc = time.time()

# Print results
print 'Time to run: ', toc - tic
print 'Optimized turbine x-coordinates:', prob['turbineX']
print 'Optimized turbine y-coordinates:', prob['turbineY']
print 'Optimized AEP:', prob['AEP']
