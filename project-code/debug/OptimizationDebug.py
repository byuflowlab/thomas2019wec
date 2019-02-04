"""
Filename:       OptimizationDebug.py
Created:        Jan. 14, 2019
Author:         Spencer McOmber
Description:    This file was created because our optimization results for our comparison study are returning odd
results. It appears the results for AEP Improvement with and without WEC are identical when the Jensen and FLORIS
turbine wake models are used. The results for the Bastankhah and Porte-Agel (BPA) turbine wake model with WEC appear
odd too--the spread of the box and whisker plot is GREATER than it is without WEC, which is the opposite of what
Thomas and Ning observed in Thomas and Ning 2018. In summary, it appears that WEC is having no effect on optimization
results for Jensen and FLORIS, and WEC is having an adverse effect on BPA.

This file is being written to test the turbine wake models to see if WEC is being applied/used correctly. A simple
4-turbine case will be considered to see if the wake models are being set up and used properly.

This file is basically a run script that will run on this local lab computer. The run script will be based on the
opt_snopt_relax.py file found under "JENSEN_wec_opt".
"""

# Copied over all of the import statements the
from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver, view_connections, SqliteRecorder
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy import config
from plantenergy.utilities import sunflower_points
from plantenergy.GeneralWindFarmComponents import calculate_distance

from scipy.interpolate import UnivariateSpline

import time
import numpy as np
import matplotlib.pyplot as plt

# import cProfile
import sys


# If this file is called directly, run the code below.
if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI:  # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

        print("In MPI, impl = ", impl)

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl


    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)


    prob = Problem(impl=impl)

    #########################################################################

    # Layout number and run number.
    input = 0
    run_number = input

    # Give path to input directory. Note that this run script that's being run is inside the 'debug' folder.
    input_directory = '../input_files/'

    # Select turbine wake model.
    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    FLORIS = 0
    BPA = 1
    JENSEN = 2
    LARSEN = 3
    model = JENSEN
    print(MODELS[model])

    # Select optimization approach/method.
    opt_algorithm = 'snopt'  # can be 'ga', 'ps', 'snopt'

    # Tell code whether or not WEC is being used with a boolean variable "relax".
    # relax = True
    relax = False

    # Take specific actions depending on whether or not WEC is being used.
    if relax:

        # Change output directory based on whether or not WEC is being used.
        output_directory = './output_files_%s_wec/' % opt_algorithm

        # Use a vector of expansion factors if WEC is being used.
        expansion_factors = np.array([3.0, 2.75, 2.50, 2.25, 2.0, 1.75, 1.50, 1.25, 1.0, 1.0])

    # Take other actions if WEC is not being used.
    else:

        output_directory = './output_files_%s/' % opt_algorithm

        # Use a vector of expansion factors if WEC is being used.
        expansion_factors = np.array([1.0])

    # Create output directory if it does not exist yet.
    import distutils.dir_util
    distutils.dir_util.mkpath(output_directory)

    # Define which wind rose data file to use.
    wind_rose_file = 'nantucket'  # can be one of: 'amalia', 'nantucket', 'directional

    # Define the air density
    air_density = 1.225  # kg/m^3 (from Jen)

    # Define turbine type.
    turbine_type = 'NREL5MW'    # can be 'V80' or 'NREL5MW'

    # Define turbine properties depending on the turbine_type that was previously defined.
    if turbine_type == 'V80':

        # define turbine size
        rotor_diameter = 80.  # (m)
        hub_height = 70.0

        z_ref = 80.0 #m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 4.  # m/s
        rated_power = 2000.  # kW
        generator_efficiency = 0.944

        ct_curve_data = np.loadtxt(input_directory + 'mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")
        ct_curve_wind_speed = ct_curve_data[:, 0]
        ct_curve_ct = ct_curve_data[:, 1]

        # air_density = 1.1716  # kg/m^3
        Ar = 0.25 * np.pi * rotor_diameter ** 2
        # cp_curve_wind_speed = ct_curve[:, 0]
        power_data = np.loadtxt(input_directory + 'niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
        # cp_curve_cp = niayifar_power_model(cp_curve_wind_speed)/(0.5*air_density*cp_curve_wind_speed**3*Ar)
        cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
        cp_curve_wind_speed = power_data[:, 0]
        cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.0001)

    elif turbine_type == 'NREL5MW':

        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.0

        z_ref = 80.0 # m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 3.  # m/s
        rated_power = 5000.  # kW
        generator_efficiency = 0.944

        filename = input_directory + "NREL5MWCPCT_dict.p"
        # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
        import cPickle as pickle

        data = pickle.load(open(filename, "rb"))
        ct_curve = np.zeros([data['wind_speed'].size, 2])
        ct_curve_wind_speed = data['wind_speed']
        ct_curve_ct = data['CT']

        # cp_curve_cp = data['CP']
        # cp_curve_wind_speed = data['wind_speed']

        loc0 = np.where(data['wind_speed'] < 11.55)
        loc1 = np.where(data['wind_speed'] > 11.7)

        cp_curve_cp = np.hstack([data['CP'][loc0], data['CP'][loc1]])
        cp_curve_wind_speed = np.hstack([data['wind_speed'][loc0], data['wind_speed'][loc1]])
        cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.000001)
    else:
        raise ValueError("Turbine type is undefined.")

    # Define the x/ro ratio.
    x_over_ro = 10.0

    # Define initial turbine coordinates. First and second turbines are the upwind turbines, third turbine is the
    # downwind turbine.
    turbineXInitialPosition = 0.0
    turbineYInitialPosition = -200.0

    secondTurbineXInitialPosition = 50.0
    secondTurbineYInitialPosition = 100.0

    thirdTurbineXInitialPosition = x_over_ro * (rotor_diameter/2.0)
    thirdTurbineYInitialPosition = -200.0

    turbineX = np.array([turbineXInitialPosition, secondTurbineXInitialPosition, thirdTurbineXInitialPosition])
    turbineY = np.array([turbineYInitialPosition, secondTurbineYInitialPosition, thirdTurbineYInitialPosition])

    turbineXInit = np.copy(turbineX)
    turbineYInit = np.copy(turbineY)

    # TODO: Define turbineX vector such that we have the same layout that was used to obtain the AEP vs. crosswind
    # position curves. Put upper and lower bounds on the upwind turbines such that they don't move (their coordinates
    # are fixed). Only things that should change are the coordinates of the downwind turbine. This way, we can test
    # if WEC is working by seeing where the downwind turbine ends up -- if wec doesn't work, then the turbine will
    # end up at the local min position; otherwise, the turbine will end up on the left or the right of the two upwind
    # turbines.

    nTurbines = turbineX.size

    # create boundary specifications
    boundary_radius = 0.5 * (rotor_diameter * 4000. / 126.4 - rotor_diameter)  # 1936.8
    center = np.array([boundary_radius, boundary_radius]) + rotor_diameter / 2.
    start_min_spacing = 5.
    nVertices = 1
    boundary_center_x = center[0]
    boundary_center_y = center[1]
    xmax = np.max(turbineX)
    ymax = np.max(turbineY)
    xmin = np.min(turbineX)
    ymin = np.min(turbineY)
    boundary_radius_plot = boundary_radius + 0.5 * rotor_diameter

    # initialize input variable arrays
    nTurbs = nTurbines
    rotorDiameter = np.zeros(nTurbs)
    hubHeight = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.  # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter  # m
        hubHeight[turbI] = hub_height  # m
        axialInduction[turbI] = 1.0 / 3.0
        Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
        # print(Ct)
        Cp[turbI] = 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
        generatorEfficiency[turbI] = generator_efficiency
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    nDirections = 1
    wind_speed = 8.1                                # m/s
    # wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_direction = 270.0       # deg (N = 0 deg., using direction FROM, as in met-mast data)
    windDirections = np.array([wind_direction])
    size = windDirections.size
    wind_frequency = 1.                             # probability of wind in this direction at this speed

    if MODELS[model] == 'JENSEN':
        # set appropriate wake model options
        wake_model_options = {'variant': 'Cosine'}
        # initialize problem
        prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
                                              wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                                              params_IdepVar_func=add_jensen_params_IndepVarComps,
                                              params_IndepVar_args={}))

    prob.driver = pyOptSparseDriver()

    if opt_algorithm == 'snopt':
        # set up optimizer
        prob.driver.options['optimizer'] = 'SNOPT'
        # prob.driver.options['gradient method'] = 'snopt_fd'

        # set optimizer options
        prob.driver.opt_settings['Verify level'] = 0
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
        prob.driver.opt_settings[
            'Print file'] = output_directory + 'SNOPT_print_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)
        prob.driver.opt_settings[
            'Summary file'] = output_directory + 'SNOPT_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)

        prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-2,
                                   active_tol=(2. * rotor_diameter) ** 2)
        prob.driver.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-2,
                                   active_tol=2. * rotor_diameter)

    prob.driver.add_objective('obj', scaler=1E-3)

    # select design variables
    prob.driver.add_desvar('turbineX', scaler=1E1, lower=np.zeros(nTurbines),
                           upper=np.ones(nTurbines) * 3. * boundary_radius)
    prob.driver.add_desvar('turbineY', scaler=1E1, lower=np.zeros(nTurbines),
                           upper=np.ones(nTurbines) * 3. * boundary_radius)
    # prob.driver.add_desvar('turbineX1', scaler=1E1, lower=turbineXInitialPosition,
    #                        upper=turbineXInitialPosition)
    # prob.driver.add_desvar('turbineY1', scaler=1E1, lower=turbineYInitialPosition,
    #                        upper=turbineYInitialPosition)
    # prob.driver.add_desvar('turbineX2', scaler=1E1, lower=secondTurbineXInitialPosition,
    #                        upper=secondTurbineXInitialPosition)
    # prob.driver.add_desvar('turbineY2', scaler=1E1, lower=secondTurbineYInitialPosition,
    #                        upper=secondTurbineYInitialPosition)
    # prob.driver.add_desvar('turbineX3', scaler=1E1, lower=secondTurbineXInitialPosition,
    #                        upper=secondTurbineXInitialPosition)
    # prob.driver.add_desvar('turbineY3', scaler=1E1, lower=np.ones(nTurbines) * -3. * boundary_radius,
    #                        upper=np.ones(nTurbines) * 3. * boundary_radius)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.root.ln_solver.options['mode'] = 'rev'

    # Begin setting up the problem and running it.
    print("almost time for setup")
    tic = time.time()
    print("entering setup at time = ", tic)
    prob.setup(check=True)
    toc = time.time()
    mpi_print(prob, "setup complete at time = ", toc)

    # print the results
    mpi_print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    # prob['turbineX1'] = turbineXInitialPosition
    # prob['turbineY1'] = turbineYInitialPosition
    # prob['turbineX2'] = secondTurbineXInitialPosition
    # prob['turbineY2'] = secondTurbineYInitialPosition
    # prob['turbineX3'] = thirdTurbineXInitialPosition
    # prob['turbineY3'] = thirdTurbineYInitialPosition
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['hubHeight'] = hubHeight
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = np.array([wind_frequency])
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_wind_speed
    cutInSpeeds = np.ones(nTurbines) * cut_in_speed
    prob['cut_in_speed'] = cutInSpeeds
    ratedPowers = np.ones(nTurbines) * rated_power
    prob['rated_power'] = ratedPowers

    # assign boundary values
    prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    prob['boundary_radius'] = boundary_radius

    # Get the initial AEP before optimizing. This will be used to see how much the AEP improved using the optimization.
    prob.run_once()
    AEP_init_opt = prob['AEP']

    # Begin loop to optimize AEP for each expansion factor.
    for expansion_factor, i in zip(expansion_factors, np.arange(0, expansion_factors.size)):

        # Pass the current relaxation factor to the problem.
        prob['model_params:relaxationFactor'] = expansion_factor

        # run the problem
        mpi_print(prob, 'start %s run' % (MODELS[model]))
        prob.run()
        mpi_print(prob, 'end %s run' % (MODELS[model]))

    # Save the most recent AEP result.
    AEP_run_opt = prob['AEP']

    # If ... some sort of condition is met ... then print all the results.
    if prob.root.comm.rank == 0:

        mpi_print(prob, 'turbine X positions in wind frame (m): %s' % prob['turbineX'])
        mpi_print(prob, 'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
        mpi_print(prob, 'wind farm power in each direction (kW): %s' % prob['dirPowers'])
        mpi_print(prob, 'Initial AEP (kWh): %s' % AEP_init_opt)
        mpi_print(prob, 'Final AEP (kWh): %s' % AEP_run_opt)
        mpi_print(prob, 'AEP improvement: %s' % (AEP_run_opt / AEP_init_opt))

    turbineX = prob['turbineX']
    turbineY = prob['turbineY']

    # Plot the resulting turbine locations.
    plt.figure(1)
    plt.plot(turbineXInit, turbineYInit, 'r*', label='Initial')
    plt.plot(turbineX, turbineY, 'b^', label='Final')
    plt.xlabel('X Coordinates (m)')
    plt.ylabel('Y Coordinates (m)')
    plt.show()
