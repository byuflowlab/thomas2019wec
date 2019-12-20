from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver, view_connections, SqliteRecorder
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy import config
# from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.utilities import sunflower_points
from plantenergy.GeneralWindFarmComponents import calculate_distance

from scipy.interpolate import UnivariateSpline

import time
import numpy as np
import matplotlib.pyplot as plt

# import cProfile
import sys


def plot_round_farm(turbineX, turbineY, rotor_diameter, boundary_center, boundary_radius, min_spacing=2.,
                    save_start=False, show_start=False, save_file=None):
    boundary_center_x = boundary_center[0]
    boundary_center_y = boundary_center[1]

    boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')

    fig, ax = plt.subplots()
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    ax.add_patch(boundary_circle)
    plt.axis('equal')
    ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
    ax.set_xlim([(boundary_center_x - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_x + boundary_radius) / rotor_diameter + 1.])
    ax.set_ylim([(boundary_center_y - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_y + boundary_radius) / rotor_diameter + 1.])

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing))
        else:
            plt.savefig(save_file)
    if show_start:
        plt.show()


def plot_square_farm(turbineX, turbineY, rotor_diameter, boundary_x, boundary_y, boundary_width, min_spacing=2,
                     save_start=False, show_start=False, save_file=None):
    full_bound_x = np.array([boundary_x[0], boundary_x[0], boundary_x[1], boundary_x[1], boundary_x[0]])
    full_bound_y = np.array([boundary_y[0], boundary_y[1], boundary_y[1], boundary_y[0], boundary_y[0]])

    real_bound_x = np.array(
        [boundary_x[0] + rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2., boundary_x[1] - rotor_diameter / 2.,
         boundary_x[1] - rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2.])
    real_bound_y = np.array(
        [boundary_y[0] + rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2.,
         boundary_y[0] + rotor_diameter / 2., boundary_y[0] + rotor_diameter / 2.])

    fig, ax = plt.subplots()
    i = 0
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        # print("here")
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
        ax.text(x, y,'%i' % i)
        i += 1
    ax.plot(full_bound_x / rotor_diameter, full_bound_y / rotor_diameter)
    ax.plot(real_bound_x / rotor_diameter, real_bound_y / rotor_diameter, '--')
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    # ax.add_patch(boundary_circle)
    plt.axis('equal')
    # ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing))
        else:
            plt.savefig(save_file)
    if show_start:
        plt.show()


if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.utils import mpi as MPI

    # if MPI:  # pragma: no cover
    #     # if you called this script with 'mpirun', then use the petsc data passing
    #     from openmdao.core.petsc_impl import PetscImpl as impl
    # 
    #     print("In MPI, impl = ", impl)
    # 
    # else:
    #     # if you didn't use 'mpirun', then use the numpy data passing
    #     from openmdao.api import BasicImpl as impl
    # 
    # 
    # def print(*args):
    #     """ helper function to only print on rank 0 """
    #     if prob.root.comm.rank == 0:
    #         print(*args)
    # 
    # 
    # prob = Problem(impl=impl)

    #########################################################################

    # set up this run

    # specify which starting layout should be used
    # layout_number = int(sys.argv[1])
    layout_number = 0
    # wec_method_number = int(sys.argv[2])
    wec_method_number = 1
    # model = int(sys.argv[3])
    model = 1
    # opt_alg_number = int(sys.argv[4])
    opt_alg_number = 0



    run_number = layout_number

    # set model
    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    print(MODELS[model])

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

    if wec_method == 'hybrid':
        WECH = 1
    else:
        WECH = 0

    differentiable = True

    turb = 6
    res = 500

    if wec_method == 'diam':
        # expansion_factors = np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
        expansion_factors = np.exp(np.arange(0, 3, .5)[::-1])
        expansion_factors = np.array([1., 1.5, 2., 2.5, 3., 3.5, 4.0])
        print(expansion_factors)
        # quit()

    elif wec_method == 'hybrid':
        # expansion_factors = np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
        expansion_factors = np.array([1., 1.5, 2., 2.5, 3., 3.5, 4.0])
        print(expansion_factors)
        # quit()

    elif wec_method == 'angle':
        # expansion_factors = 10*np.array([3, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 1.0])
        # expansion_factors = np.array([60., 50., 40., 30., 20., 10., 0.0, 0.0])
        expansion_factors = np.array([60., 0.0])
    # for expansion_factor in np.array([5., 4., 3., 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]):
    # for expansion_factor in np.array([20., 15., 10., 5., 4., 3., 2.5, 1.25, 1.0]):
    # expansion_factors = np.array([20., 10., 5., 2.5, 1.25, 1.0])
    print(expansion_factors)
    wake_combination_method = 1  # can be [0:Linear freestreem superposition,
    #  1:Linear upstream velocity superposition,
    #  2:Sum of squares freestream superposition,
    #  3:Sum of squares upstream velocity superposition]

    ti_calculation_method = 0  # can be [0:No added TI calculations,
    # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
    # 2:TI by Niayifar and Porte Agel 2016,
    # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
    # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
    # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    ti_opt_method = 0  # can be [0:No added TI calculations,
    # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
    # 2:TI by Niayifar and Porte Agel 2016,
    # 3:TI by Niayifar and Porte Agel 2016 with added soft max function,
    # 4:TI by Niayifar and Porte Agel 2016 using area overlap ratio,
    # 5:TI by Niayifar and Porte Agel 2016 using area overlap ratio and SM function]

    final_ti_opt_method = 5

    sm_smoothing = 700.

    if ti_calculation_method == 0:
        calc_k_star_calc = False
    else:
        calc_k_star_calc = True

    if ti_opt_method == 0:
        calc_k_star_opt = False
    else:
        calc_k_star_opt = True

    nRotorPoints = 100

    wind_rose_file = 'amalia'  # can be one of: 'amalia', 'nantucket', 'directional

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

        z_ref = 80.0 #m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 4.  # m/s
        # cut_in_speed = 0.0
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

    layout_data = np.loadtxt(layout_directory + "layouts/grid_16turbs/nTurbs16_spacing5_layout_%i.txt" % layout_number)
    # layout_data = np.loadtxt(layout_directory+"layouts/nTurbs9_spacing5_layout_%i.txt" % layout_number)

    turbineX = np.copy(layout_data[:, 0] * rotor_diameter)
    turbineY = np.copy(layout_data[:, 1] * rotor_diameter)

    turbineX_init = np.copy(turbineX)
    turbineY_init = np.copy(turbineY)

    nTurbines = turbineX.size

    boundary_x = np.array([0.0, 5. * rotor_diameter * (np.sqrt(nTurbines) - 1) + rotor_diameter])
    boundary_y = np.array([0.0, 5. * rotor_diameter * (np.sqrt(nTurbines) - 1) + rotor_diameter])

    plot_square_farm(turbineX_init, turbineY_init, rotor_diameter, boundary_x, boundary_y, boundary_x[1] - boundary_x[0],
                     show_start=show_start)

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
    if wind_rose_file is 'nantucket':
        # windRose = np.loadtxt(input_directory + 'nantucket_windrose_ave_speeds.txt')
        windRose = np.loadtxt(input_directory + 'nantucket_wind_rose_for_LES.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file is 'amalia':
        windRose = np.loadtxt(input_directory + 'windrose_amalia_directionally_averaged_speeds.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file is 'directional':
        windRose = np.loadtxt(input_directory + 'directional_windrose.txt')
        windDirections = windRose[:, 0]
        windSpeeds = windRose[:, 1]
        windFrequencies = windRose[:, 2]
        size = np.size(windDirections)
    elif wind_rose_file is 'uno':
        windDirections = np.array([270.])
        windSpeeds = np.array([8.0])
        windFrequencies = np.array([1.0])
        size = np.size(windDirections)
    else:
        size = 20
        windDirections = np.linspace(0, 270, size)
        windFrequencies = np.ones(size) / size
    # size=1
    # windDirections = np.array([270])
    # windFrequencies = np.array([1])
    # windSpeeds = np.array([8])
    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speed': ct_curve_wind_speed,
                          'interp_type': 1,
                          'use_rotor_components': False,
                          'differentiable': differentiable,
                          'verbose': False}

    nVertices = 0

    if MODELS[model] == 'BPA':
        # initialize problem
        prob = Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=differentiable,
                                              use_rotor_components=False,
                                              wake_model=gauss_wrapper,
                                              params_IdepVar_func=add_gauss_params_IndepVarComps,
                                              params_IdepVar_args={'nRotorPoints': nRotorPoints},
                                              wake_model_options=wake_model_options,
                                              cp_points=cp_curve_cp.size, cp_curve_spline=cp_curve_spline,
                                              record_function_calls = rec_func_calls))

    elif MODELS[model] == 'FLORIS':
        # initialize problem
        prob = Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
                                              minSpacing=minSpacing, differentiable=differentiable,
                                              use_rotor_components=False,
                                              wake_model=floris_wrapper,
                                              params_IdepVar_func=add_floris_params_IndepVarComps,
                                              params_IdepVar_args={}, record_function_calls = rec_func_calls))
    # elif MODELS[model] == 'JENSEN':
    #     initialize problem
    # prob = Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
    #                                       minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
    #                                       wake_model=jensen_wrapper,
    #                                       params_IdepVar_func=add_jensen_params_IndepVarComps,
    #                                       params_IdepVar_args={}, record_function_calls = rec_func_calls))
    else:
        ValueError('The %s model is not currently available. Please select BPA or FLORIS' % (MODELS[model]))
    # prob.model.deriv_options['type'] = 'fd'
    # prob.model.deriv_options['form'] = 'central'
    # prob.model.deriv_options['step_size'] = 1.0e-8
    # from openmdao.api import DirectSolver
    # prob.model.linear_solver = DirectSolver()

    print("almost time for setup")
    tic = time.time()
    print("entering setup at time = ", tic)
    prob.setup(check=False)
    toc = time.time()
    print("setup complete at time = ", toc)

    # print the results
    print(('Problem setup took %.03f sec.' % (toc - tic)))

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
    prob['rated_wind_speed'] = 16.*np.ones(nTurbines)
    prob['use_power_curve_definition'] = True

    # assign boundary values
    # prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    # prob['boundary_radius'] = boundary_radius

    if MODELS[model] is 'BPA':
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

    prob.run_model()
    AEP_init_calc = np.copy(prob['AEP'])
    print(AEP_init_calc * 1E-6)

    if MODELS[model] is 'BPA':
        prob['model_params:ti_calculation_method'] = np.copy(ti_opt_method)
        prob['model_params:calc_k_star'] = np.copy(calc_k_star_opt)

    prob.run_model()
    AEP_init_opt = np.copy(prob['AEP'])
    AEP_run_opt = np.copy(AEP_init_opt)
    print(AEP_init_opt * 1E-6)

    config.obj_func_calls_array[:] = 0.0
    config.sens_func_calls_array[:] = 0.0

    expansion_factor_last = 0.0

    tict = time.time()


    xsweep = np.linspace(np.min(turbineX_init), np.max(turbineX_init), res)
    ysweep = np.linspace(np.min(turbineY_init), np.max(turbineY_init), res)

    xaep = np.zeros((expansion_factors.size, xsweep.size))
    yaep = np.zeros((expansion_factors.size, ysweep.size))
    a45aep = np.zeros((expansion_factors.size, xsweep.size))
    xvel = np.zeros((expansion_factors.size, xsweep.size))
    xpow = np.zeros((expansion_factors.size, xsweep.size))
    yvel = np.zeros((expansion_factors.size, ysweep.size))
    ypow = np.zeros((expansion_factors.size, ysweep.size))

    xfig, xax = plt.subplots(1,3, figsize=(12, 5))
    yfig, yax = plt.subplots(1,3, figsize=(12, 5))
    # a45fig, a45ax = plt.subplots()

    for expansion_factor, i in zip(expansion_factors, np.arange(0, expansion_factors.size)):  # best so far
        # print("func calls: ", config.obj_func_calls_array, np.sum(config.obj_func_calls_array))
        # print("grad func calls: ", config.sens_func_calls_array, np.sum(config.sens_func_calls_array))
        # AEP_init_run_opt = prob['AEP']

        print("starting run with exp. fac = ", expansion_factor)

        prob['turbineX'] = np.copy(turbineX_init)
        prob['turbineY'] = np.copy(turbineY_init)

        if MODELS[model] is 'BPA':
            prob['model_params:ti_calculation_method'] = np.copy(ti_opt_method)
            prob['model_params:calc_k_star'] = np.copy(calc_k_star_opt)
            if wec_method == 'diam' or wec_method == 'hybrid':
                prob['model_params:wec_factor'] = np.copy(expansion_factor)
            elif wec_method == 'angle':
                prob['model_params:wec_spreading_angle'] = np.copy(expansion_factor)

        # run the problem
        print('start %s run' % (MODELS[model]))

        prob['turbineY'][turb] += 2.*rotor_diameter
        for xloc in np.arange(0, res):
            prob['turbineX'][turb] = xsweep[xloc]
            prob.run_model()
            xaep[i, xloc] = np.copy(prob['AEP'])
            xvel[i, xloc] = np.copy(prob['wtVelocity0'][turb])
            xpow[i, xloc] = np.copy(prob['wtPower0'][turb])
            # xaep[i, xloc] = np.copy(prob['wtPower0'][turb])

        prob['turbineX'] = np.copy(turbineX_init)
        prob['turbineY'] = np.copy(turbineY_init)
        prob['turbineX'][turb] += 2.*rotor_diameter
        for yloc in np.arange(0, res):
            prob['turbineY'][turb] = ysweep[yloc]
            prob.run_model()
            yaep[i, yloc] = np.copy(prob['AEP'])
            yvel[i, yloc] = np.copy(prob['wtVelocity0'][turb])
            ypow[i, yloc] = np.copy(prob['wtPower0'][turb])
            # yaep[i, yloc] = np.copy(prob['wtPower0'][turb])

        # for a45loc in np.arange(0, res):
        #     prob['turbineX'][turb] = xsweep[a45loc]
        #     prob['turbineY'][turb] = ysweep[a45loc]
        #     prob.run_model()
        #     a45aep[i, a45loc] = np.copy(prob['AEP'])

        # plot xsweep
        xax[0].plot(xsweep, -xaep[i][:]*1E-6, label='AEP %s %.3f' % (wec_method, expansion_factor))
        xax[1].plot(xsweep, -xvel[i][:], label='Vel. %s %.3f' % (wec_method, expansion_factor))
        xax[2].plot(xsweep, -xpow[i][:], label='Pow. %s %.3f' % (wec_method, expansion_factor))
        # plot ysweep
        yax[0].plot(ysweep, -yaep[i][:]*1E-6, label='AEP %s %.3f' % (wec_method, expansion_factor))
        yax[1].plot(ysweep, -yvel[i][:], label='Vel. %s %.3f' % (wec_method, expansion_factor))
        yax[2].plot(ysweep, -ypow[i][:], label='Pow. %s %.3f' % (wec_method, expansion_factor))

        # plot ysweep
        # a45ax.plot(np.hypot(xsweep,ysweep), -a45aep[i][:]*1E-6, label='%s %.3f' % (wec_method, expansion_factor))



    np.savetxt(input_directory + 'sweep_coords.txt', np.c_[xsweep, ysweep],
               header='x, y')

    np.savetxt(input_directory + 'sweep_xaep.txt', xaep, header='rows: wec factors, cols:xlocs')
    np.savetxt(input_directory + 'sweep_yaep.txt', yaep, header='rows: wec factors, cols:ylocs')

    xax[0].legend()
    # xax[1].legend()
    yax[0].legend()
    # yax[1].legend()
    # a45ax.legend()

    xax[0].set_xlabel('Turbine %i, Downstream Location' % turb)
    xax[1].set_xlabel('Turbine %i, Downstream Location' % turb)
    xax[2].set_xlabel('Turbine %i, Downstream Location' % turb)
    yax[0].set_xlabel('Turbine %i, Crosswind Location' % turb)
    yax[1].set_xlabel('Turbine %i, Crosswind Location' % turb)
    yax[2].set_xlabel('Turbine %i, Crosswind Location' % turb)
    # a45ax.set_xlabel('Turbine %i, distance from Origin' % turb)

    xax[0].set_ylabel('AEP (GWh)')
    xax[1].set_ylabel('Vel. (m/s)')
    xax[2].set_ylabel('Pow. (kW)')
    yax[0].set_ylabel('AEP (GWh)')
    yax[1].set_ylabel('Vel. (m/s)')
    yax[2].set_ylabel('Pow. (kW)')
    # a45ax.set_ylabel('AEP (GWh)')

    xax[0].set_title('X Sweep')
    yax[0].set_title('Y Sweep')
    # a45ax.set_title('45 deg. Angle Sweep')
    plt.tight_layout()
    plt.show()