import numpy as np 
import regex as re

import openmdao.api as om
from openmdao.devtools import iprofile

from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy import config
from plantenergy.utilities import sunflower_points
from plantenergy.GeneralWindFarmComponents import calculate_distance

from scipy.interpolate import UnivariateSpline

import time
import numpy as np
import matplotlib.pyplot as plt

# import cProfile
import sys
import os

def parse_alpso_file(path, nturbines):

    fcall_pattern = "(?<=NUMBER OF OBJECTIVE FUNCTION EVALUATIONS: ).*"
    obj_pattern = "(?<=	F = -).*"
 
    # parse file for objective values
    # with open(path) as f:
    #     obj = list(map(np.float, re.findall(obj_pattern,f.read(),re.MULTILINE)))

    # parse file for function calls
    with open(path) as f:
        fcalls = list(map(np.int, re.findall(fcall_pattern,f.read(),re.MULTILINE)))

    # find how many steps in optimization
    print(fcalls)
    steps = len(fcalls)

    # initialize position array with correct shape steps by turbines
    xpositions = np.zeros([steps, nturbines])
    ypositions = np.zeros([steps, nturbines])

    # parse file for all turbine x locations at each step
    for i in np.arange(0, nturbines):
        with open(path) as f:
            # find the location fot he given wind turbine at all steps
            pattern = re.compile('(?<=P\\(%i\\) = )\d+.\d+|(?<=P\\(%i\\) = )\d' %(i,i))
            p = re.findall(pattern, f.read())

        # save positions for this turbine at all steps
        if len(p) < steps:
            print("error at x ", i)
        xpositions[:, i] = p

    # parse file for all turbine y locations at each step
    for i in np.arange(nturbines, 2*nturbines):
        with open(path) as f:
            # find the location fot he given wind turbine at all steps
            pattern = re.compile('(?<=P\\(%i\\) = )\d+.\d+|(?<=P\\(%i\\) = )\d' %(i,i))
            p = re.findall(pattern, f.read())

        # save positions for this turbine at all steps
        if len(p) < steps:
            print("error at y ", i)
        ypositions[:, i-nturbines] = p

    return fcalls, xpositions*1E4, ypositions*1E4

def set_up_prob():
    layout_number = 0
    wec_method_number = 0
    wake_model = 1
    opt_alg_number = 2
    max_wec = 3
    nsteps = 6
    record = False
    OPENMDAO_REQUIRE_MPI = False
    run_number = layout_number
    model = wake_model
    # set model
    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    # print(MODELS[model])

    # select optimization approach/method
    opt_algs = ['snopt', 'ga', 'ps']
    opt_algorithm = opt_algs[opt_alg_number]

    # select wec method
    wec_methods = ['none', 'diam', 'angle', 'hybrid']
    wec_method = wec_methods[wec_method_number]

    # pop_size = 760

    # save and show options
    show_start = False
    show_end = False
    save_start = False
    save_end = False

    save_locations = True
    save_aep = True
    save_time = True
    rec_func_calls = True

    input_directory = "../../../input_files/"

    # set options for BPA
    print_ti = False
    sort_turbs = True

    # turbine_type = 'NREL5MW'            #can be 'V80' or 'NREL5MW'
    turbine_type = 'V80'  # can be 'V80' or 'NREL5MW'

    wake_model_version = 2016

    WECH = 0
    if wec_method == 'diam':
        output_directory = "../output_files/%s_wec_diam_max_wec_%i_nsteps_%.3f/" % (opt_algorithm, max_wec, nsteps)
        relax = True
        # expansion_factors = np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])

        expansion_factors = np.linspace(1.0, max_wec, nsteps)
        if opt_algorithm == 'ps':
            expansion_factors = np.append(np.flip(expansion_factors), 1.0)
        else:
            expansion_factors = np.append(np.flip(expansion_factors), 1.0)
        conv_tols = np.array([9E-3, 9E-3, 9E-3, 9E-3, 9E-3, 9E-3, 1E-3])
    elif wec_method == 'angle':
        output_directory = "../output_files/%s_wec_angle_max_wec_%i_nsteps_%.3f/" % (opt_algorithm, max_wec, nsteps)
        relax = True
        # expansion_factors = np.array([50, 40, 30, 20, 10, 0.0, 0.0])
        expansion_factors = np.linspace(0.0, max_wec, nsteps)
        expansion_factors = np.append(np.flip(expansion_factors), 0.0)
    elif wec_method == 'hybrid':
        expansion_factors = np.linspace(1.0, max_wec, nsteps)
        expansion_factors = np.append(np.flip(expansion_factors), 1.0)
        output_directory = "../output_files/%s_wec_hybrid_max_wec_%i_nsteps_%.3f/" % (opt_algorithm, max_wec, nsteps)
        relax = True
        WECH = 1
    elif wec_method == 'none':
        relax = False
        if opt_algorithm == 'ps':
            expansion_factors = np.array([1.0])
            conv_tols = np.array([1E-4])
        else:
            expansion_factors = np.array([1.0, 1.0])
            conv_tols = np.array([9E-3, 1E-3])
        output_directory = "../output_files/%s/" % opt_algorithm
    else:
        raise ValueError('wec_method must be diam, angle, hybrid, or none')

    # create output directory if it does not exist yet
    import distutils.dir_util
    distutils.dir_util.mkpath(output_directory)

    differentiable = True

    # for expansion_factor in np.array([5., 4., 3., 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]):
    # for expansion_factor in np.array([20., 15., 10., 5., 4., 3., 2.5, 1.25, 1.0]):
    # expansion_factors = np.array([20., 10., 5., 2.5, 1.25, 1.0])

    wake_combination_method = 1  # can be [0:Linear freestreem superposition,
    #  1:Linear upstream velocity superposition,
    #  2:Sum of squares freestream superposition,
    #  3:Sum of squares upstream velocity superposition]

    ti_calculation_method = 4  # can be [0:No added TI calculations,
    # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
    # 2:TI by Niayifar and Porte Agel 2016,
    # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
    # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
    # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    if wec_method_number > 0:
        ti_opt_method = 0  # can be [0:No added TI calculations,
        # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
        # 2:TI by Niayifar and Porte Agel 2016,
        # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
        # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
        # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    else:
        ti_opt_method = 0
    final_ti_opt_method = 5

    if opt_algorithm == 'ps' and wec_method == 'none':
        ti_opt_method = ti_calculation_method

    sm_smoothing = 700.

    if ti_calculation_method == 0:
        calc_k_star_calc = False
    else:
        calc_k_star_calc = True

    if ti_opt_method == 0:
        calc_k_star_opt = False
    else:
        calc_k_star_opt = True

    nRotorPoints = 1

    wind_rose_file = 'nantucket'  # can be one of: 'amalia', 'nantucket', 'directional

    TI = 0.108
    k_calc = 0.3837 * TI + 0.003678
    # k_calc = 0.022
    # k_opt = 0.04

    shear_exp = 0.31

    # air_density = 1.1716  # kg/m^3
    air_density = 1.225  # kg/m^3 (from Jen)

    if turbine_type == 'V80':

        # define turbine size
        rotor_diameter = 80.  # (m)
        hub_height = 70.0

        z_ref = 80.0  # m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 4.  # m/s
        cut_out_speed = 25.  # m/s
        rated_wind_speed = 16.  # m/s
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

        z_ref = 80.0  # m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 3.  # m/s
        cut_out_speed = 25.  # m/s
        rated_wind_speed = 11.4  # m/s
        rated_power = 5000.  # kW
        generator_efficiency = 0.944

        filename = input_directory + "NREL5MWCPCT_dict.p"
        # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
        import pickle

        data = pickle.load(open(filename, "rb"), encoding='latin1')
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

    # load starting locations
    layout_directory = input_directory

    layout_data = np.loadtxt(layout_directory + "layouts/round_38turbs/nTurbs38_spacing5_layout_%i.txt" % layout_number)
    # layout_data = np.loadtxt(layout_directory + "layouts/grid_16turbs/nTurbs16_spacing5_layout_%i.txt" % layout_number)
    # layout_data = np.loadtxt(layout_directory+"layouts/nTurbs9_spacing5_layout_%i.txt" % layout_number)

    turbineX = layout_data[:, 0] * rotor_diameter + rotor_diameter / 2.
    turbineY = layout_data[:, 1] * rotor_diameter + rotor_diameter / 2.

    turbineX_init = np.copy(turbineX)
    turbineY_init = np.copy(turbineY)

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
    if wind_rose_file == 'nantucket':
        # windRose = np.loadtxt(input_directory + 'nantucket_windrose_ave_speeds.txt')
        windRose = np.loadtxt(input_directory + 'nantucket_wind_rose_for_LES.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file == 'amalia':
        windRose = np.loadtxt(input_directory + 'windrose_amalia_directionally_averaged_speeds.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file == 'directional':
        windRose = np.loadtxt(input_directory + 'directional_windrose.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file == '1d':
        windDirections = np.array([270.])
        windSpeeds = np.array([8.0])
        windFrequencies = np.array([1.0])
        size = np.size(windDirections)
    else:
        size = 20
        windDirections = np.linspace(0, 270, size)
        windFrequencies = np.ones(size) / size

    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speed': ct_curve_wind_speed,
                          'interp_type': 1,
                          'use_rotor_components': False,
                          'differentiable': differentiable,
                          'verbose': False}

    if MODELS[model] == 'BPA':
        # initialize problem
        prob = om.Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                       minSpacing=minSpacing, differentiable=differentiable,
                                       use_rotor_components=False,
                                       wake_model=gauss_wrapper,
                                       params_IdepVar_func=add_gauss_params_IndepVarComps,
                                       params_IdepVar_args={'nRotorPoints': nRotorPoints},
                                       wake_model_options=wake_model_options,
                                       cp_points=cp_curve_cp.size, cp_curve_spline=cp_curve_spline,
                                       record_function_calls=True, runparallel=False))
    elif MODELS[model] == 'FLORIS':
        # initialize problem
        prob = om.Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                       minSpacing=minSpacing, differentiable=differentiable,
                                       use_rotor_components=False,
                                       wake_model=floris_wrapper,
                                       params_IdepVar_func=add_floris_params_IndepVarComps,
                                       params_IdepVar_args={},
                                       record_function_calls=True))
    
    else:
        ValueError('The %s model is not currently available. Please select BPA or FLORIS' % (MODELS[model]))
    
    prob.driver = om.pyOptSparseDriver()

    if opt_algorithm == 'snopt':
        # set up optimizer
        prob.driver.options['optimizer'] = 'SNOPT'
        # prob.driver.options['gradient method'] = 'snopt_fd'

        # set optimizer options
        prob.driver.opt_settings['Verify level'] = -1
        # set optimizer options
        prob.driver.opt_settings['Major optimality tolerance'] = np.float(1e-3)

        prob.driver.opt_settings[
            'Print file'] = output_directory + 'SNOPT_print_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)
        prob.driver.opt_settings[
            'Summary file'] = output_directory + 'SNOPT_summary_multistart_%iturbs_%sWindRose_%idirs_%sModel_RunID%i.out' % (
            nTurbs, wind_rose_file, size, MODELS[model], run_number)

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-4)  # ,
        # active_tol=(2. * rotor_diameter) ** 2)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-4)  # ,
        # active_tol=2. * rotor_diameter)

        prob.driver.options['dynamic_derivs_sparsity'] = True

    elif opt_algorithm == 'ga':

        prob.driver.options['optimizer'] = 'NSGA2'

        prob.driver.opt_settings['PrintOut'] = 1

        prob.driver.opt_settings['maxGen'] = 50000

        prob.driver.opt_settings['PopSize'] = 10 * nTurbines * 2

        # prob.driver.opt_settings['pMut_real'] = 0.001

        prob.driver.opt_settings['xinit'] = 1

        prob.driver.opt_settings['rtol'] = 1E-4

        prob.driver.opt_settings['atol'] = 1E-4

        prob.driver.opt_settings['min_tol_gens'] = 200

        prob.driver.opt_settings['file_number'] = run_number

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-4)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-4)


    elif opt_algorithm == 'ps':
        prob.driver.options['optimizer'] = 'ALPSO'
        prob.driver.opt_settings["SwarmSize"] = 30  # Number of Particles (Depends on Problem dimensions)
        # prob.driver.opt_settings["maxOuterIter"] = OuterIter # Maximum Number of Outer Loop Iterations (Major Iterations)
        # prob.driver.opt_settings["maxInnerIter"] = InnerIter  # Maximum Number of Inner Loop Iterations (Minor Iterations)
        # prob.driver.opt_settings["minInnerIter"] = InnerIter  # Minimum Number of Inner Loop Iterations (Dynamic Inner Iterations)
        # prob.driver.opt_settings["dynInnerIter"] = 0  # Dynamic Number of Inner Iterations Flag
        prob.driver.opt_settings["stopCriteria"] = 0  # Stopping Criteria Flag (0 - maxIters, 1 - convergence)
        prob.driver.opt_settings["stopIters"] = 5  # Consecutive Number of Iterations for which the Stopping Criteria must be Satisfied
        prob.driver.opt_settings["etol"] = 1e-3  # Absolute Tolerance for Equality constraints
        prob.driver.opt_settings["itol"] = 1e-3  # Absolute Tolerance for Inequality constraints
        # 'ltol':[float, 1e-2],            # Absolute Tolerance for Lagrange Multipliers
        prob.driver.opt_settings["rtol"] = 1e-6  # Relative Tolerance for Lagrange Multipliers
        prob.driver.opt_settings["atol"] = 1e-6  # Absolute Tolerance for Lagrange Function
        prob.driver.opt_settings["dtol"] = 1e-1  # Relative Tolerance in Distance of All Particles to Terminate (GCPSO)
        prob.driver.opt_settings["printOuterIters"] = 0  # Number of Iterations Before Print Outer Loop Information
        prob.driver.opt_settings["printInnerIters"] = 0  # Number of Iterations Before Print Inner Loop Information
        prob.driver.opt_settings["rinit"] = 1.0  # Initial Penalty Factor
        prob.driver.opt_settings["xinit"] = 1  # Initial Position Flag (0 - no position, 1 - position given)
        prob.driver.opt_settings["vinit"] = 1.0  # Initial Velocity of Particles in Normalized [-1, 1] Design Space
        prob.driver.opt_settings["vmax"] = 2.0  # Maximum Velocity of Particles in Normalized [-1, 1] Design Space
        prob.driver.opt_settings["c1"] = 2.0  # Cognitive Parameter
        prob.driver.opt_settings["c2"] = 1.0  # Social Parameter
        prob.driver.opt_settings["w1"] = 0.99  # Initial Inertia Weight
        prob.driver.opt_settings["w2"] = 0.55  # Final Inertia Weight
        prob.driver.opt_settings["ns"] = 15 # Number of Consecutive Successes in Finding New Best Position of Best Particle Before Search Radius will be Increased (GCPSO)
        prob.driver.opt_settings["nf"] = 5 # Number of Consecutive Failures in Finding New Best Position of Best Particle Before Search Radius will be Increased (GCPSO)
        prob.driver.opt_settings["dt"] = 1.0  # Time step
        prob.driver.opt_settings["vcrazy"] = 1e-2 # Craziness Velocity (Added to Particle Velocity After Updating the Penalty Factors and Langangian Multipliers)
        prob.driver.opt_settings["fileout"] = 1  # Flag to Turn On Output to filename
        # prob.driver.opt_settings["filename"] = "ALPSO.out" # We could probably remove fileout flag if filename or fileinstance is given
        prob.driver.opt_settings["seed"] = 0.0  # Random Number Seed (0 - Auto-Seed based on time clock)
        prob.driver.opt_settings["HoodSize"] = 5  # Number of Neighbours of Each Particle
        prob.driver.opt_settings["HoodModel"] = "gbest" # Neighbourhood Model (dl/slring - Double/Single Link Ring, wheel - Wheel, Spatial - based on spatial distance, sfrac - Spatial Fraction)
        prob.driver.opt_settings["HoodSelf"] = 1 # Selfless Neighbourhood Model (0 - Include Particle i in NH i, 1 - Don't Include Particle i)
        prob.driver.opt_settings["Scaling"] = 1  # Design Variables Scaling Flag (0 - no scaling, 1 - scaling between [-1, 1])
        # prob.driver.opt_settings["parallelType"] = 'EXT'  # Type of parallelization ('' or 'EXT')

        prob.model.add_constraint('sc', lower=np.zeros(int(((nTurbs - 1.) * nTurbs / 2.))), scaler=1E-7)
        prob.model.add_constraint('boundaryDistances', lower=(np.zeros(1 * turbineX.size)), scaler=1E-7)

        # prob.driver.add_objective('obj', scaler=1E0)
    prob.model.add_objective('obj', scaler=1E-9)

    # select design variables
    prob.model.add_design_var('turbineX', scaler=1E-4, lower=np.zeros(nTurbines),
                              upper=np.ones(nTurbines) * 2. * boundary_radius)
    prob.model.add_design_var('turbineY', scaler=1E-4, lower=np.zeros(nTurbines),
                              upper=np.ones(nTurbines) * 2. * boundary_radius)

    if record:
        driver_recorder = om.SqliteRecorder(output_directory + 'recorded_data_driver_%s.sql' %(run_number))
        # model_recorder = om.SqliteRecorder(output_directory + 'recorded_data_model_%s.sql' %(run_number))
        prob.driver.add_recorder(driver_recorder)
        # prob.model.add_recorder(model_recorder)
        prob.driver.recording_options['record_constraints'] = False
        prob.driver.recording_options['record_derivatives'] = False
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_inputs'] = False
        prob.driver.recording_options['record_model_metadata'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['includes'] = ['AEP']
        prob.driver.recording_options['record_responses'] = False

    # print("almost time for setup")
    tic = time.time()
    # print("entering setup at time = ", tic)
    prob.setup(check=True)
    toc = time.time()
    # print("setup complete at time = ", toc)

    # print the results
    # print(('Problem setup took %.03f sec.' % (toc - tic)))

    # assign initial values to design variables
    prob['turbineX'] = np.copy(turbineX)
    prob['turbineY'] = np.copy(turbineY)
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['hubHeight'] = hubHeight
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_wind_speed
    cutInSpeeds = np.ones(nTurbines) * cut_in_speed
    prob['cut_in_speed'] = cutInSpeeds
    ratedPowers = np.ones(nTurbines) * rated_power
    prob['rated_power'] = ratedPowers

    # assign values to turbine states
    prob['cut_in_speed'] = np.ones(nTurbines) * cut_in_speed
    prob['cut_out_speed'] = np.ones(nTurbines) * cut_out_speed
    prob['rated_power'] = np.ones(nTurbines) * rated_power
    prob['rated_wind_speed'] = np.ones(nTurbines) * rated_wind_speed
    prob['use_power_curve_definition'] = True

    # assign boundary values
    prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    prob['boundary_radius'] = boundary_radius

    if MODELS[model] == 'BPA':
        prob['model_params:wake_combination_method'] = np.copy(wake_combination_method)
        prob['model_params:ti_calculation_method'] = np.copy(ti_calculation_method)
        prob['model_params:wake_model_version'] = np.copy(wake_model_version)
        prob['model_params:wec_factor'] = 1.0
        prob['model_params:wec_spreading_angle'] = 0.0
        prob['model_params:calc_k_star'] = np.copy(calc_k_star_calc)
        prob['model_params:sort'] = np.copy(sort_turbs)
        prob['model_params:z_ref'] = np.copy(z_ref)
        prob['model_params:z_0'] = np.copy(z_0)
        prob['model_params:ky'] = np.copy(k_calc)
        prob['model_params:kz'] = np.copy(k_calc)
        prob['model_params:print_ti'] = np.copy(print_ti)
        prob['model_params:shear_exp'] = np.copy(shear_exp)
        prob['model_params:I'] = np.copy(TI)
        prob['model_params:sm_smoothing'] = np.copy(sm_smoothing)
        prob['model_params:WECH'] = WECH
        if nRotorPoints > 1:
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

        # prob.run_model()
        # AEP_init_calc = np.copy(prob['AEP'])*1E3 # convert to Whr
        # print("initial AEP: ", AEP_init_calc)
    return prob

def calculate_aep(prob, xpositions, ypositions):
    prob["turbineX"] = xpositions
    prob["turbineY"] = ypositions
    prob.run_model()
    AEP_init_calc = np.copy(prob['AEP'])*1E3 # convert to Whr
    return AEP_init_calc

# def get_position_set(directory, runid, , titypes=None):
#     if titypes == None:
#         titypes = np.zeros_like(expansion_factors)
#         titypes[-1] = 5


def get_convergencec_histories(directory, nturbines=38, nruns=200, nsteps=6, maxwec=3):

    # name output file
    output_file_pswec = directory+"convergence_histories.txt"
    # set up expansion factor array
    expansion_factors=np.append(np.flip(np.linspace(1,maxwec,nsteps)),1)
    # set up ti types array
    ti_types = np.array([0, 0, 0, 0, 0, 0, 5])

    # initialize model
    prob = set_up_prob()

    # parse each full set of optimization runs
    for i in np.arange(0, nruns):
        print("Run: %i" %(i))
        # parse the file for each step
        for j in np.arange(0, nsteps+1):
            print("Run: %i" %(i))
            print("Step: %i" %j)
            filename = "ALPSO_summary_multistart_38turbs_nantucketWindRose_12dirs_BPAModel_RunID%i_EF%.3f_TItype%i_print.out" % (i, expansion_factors[j], ti_types[j])
            path = directory+filename
            fcalls, xpositions, ypositions = parse_alpso_file(path, nturbines)
            oi = len(fcalls)
            aep = np.zeros(oi)
            for k in np.arange(0, oi):
                a = calculate_aep(prob, xpositions[k,:], ypositions[k,:])
                print(a)
                aep[k] = a[0]
            f = open(output_file_pswec, "a")
            if i == 0 and j == 0:
                header = "convergence history alternating row function calls, AEP (W)"
            else:
                header = ""
            print(fcalls, aep)
            np.savetxt(f, (fcalls, aep), header=header)
            f.close()

    return

if __name__ == "__main__":

    directory = "../output_files/pswec/"
    # directory = "../output_files/ps_wec_diam_max_wec_3_nsteps_6.000/"
    get_convergencec_histories(directory, nruns=1)

