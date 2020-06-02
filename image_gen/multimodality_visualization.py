from __future__ import print_function

import openmdao.api as om
from openmdao.devtools import iprofile

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
from mpl_toolkits import mplot3d

# import cProfile
import sys
import mpi4py.MPI


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
    # ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
    ax.set_xlim([(boundary_center_x - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_x + boundary_radius) / rotor_diameter + 1.])
    ax.set_ylim([(boundary_center_y - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_y + boundary_radius) / rotor_diameter + 1.])

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing), transparent=True)
        else:
            plt.savefig(save_file, transparent=True)
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
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        # print("here")
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
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

def run_opt(layout_number, wec_method_number, wake_model, opt_alg_number, max_wec, nsteps, xs, ys):
    OPENMDAO_REQUIRE_MPI = False
    run_number = layout_number
    model = wake_model
    # set model
    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    print(MODELS[model])

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
    save_start = True
    save_end = True

    save_locations = True
    save_aep = True
    save_time = True
    rec_func_calls = True

    input_directory = "../project-code/input_files/"

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
        expansion_factors = np.append(np.flip(expansion_factors), 1.0)
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
        expansion_factors = np.array([1.0, 1.0])
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

    if opt_algorithm == 'ps':
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
    boundary_radius_plot = boundary_radius + 0.5 * rotor_diameter

    plot_round_farm(turbineX, turbineY, rotor_diameter, [boundary_center_x, boundary_center_y], boundary_radius,
                    show_start=show_start, save_start=save_start, save_file="farm_layout.pdf")
    # quit()
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
    elif wind_rose_file is '1d':
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
    # elif MODELS[model] == 'JENSEN':
    #     initialize problem
    # prob = om.Problem(model=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=nVertices,
    #                                       minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
    #                                       wake_model=jensen_wrapper,
    #                                       params_IdepVar_func=add_jensen_params_IndepVarComps,
    #                                       params_IdepVar_args={},
    #                                               record_function_calls=True))
    else:
        ValueError('The %s model is not currently available. Please select BPA or FLORIS' % (MODELS[model]))

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

    # assign values to turbine states
    prob['cut_in_speed'] = np.ones(nTurbines) * cut_in_speed
    prob['cut_out_speed'] = np.ones(nTurbines) * cut_out_speed
    prob['rated_power'] = np.ones(nTurbines) * rated_power
    prob['rated_wind_speed'] = np.ones(nTurbines) * rated_wind_speed
    prob['use_power_curve_definition'] = True

    # assign boundary values
    prob['boundary_center'] = np.array([boundary_center_x, boundary_center_y])
    prob['boundary_radius'] = boundary_radius

    if MODELS[model] is 'BPA':
        prob['model_params:wake_combination_method'] = np.copy(wake_combination_method)
        prob['model_params:ti_calculation_method'] = np.copy(ti_opt_method)
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

    xx, yy = np.meshgrid(xs, ys)
    vel_out = np.zeros((xs.size, ys.size))
    for i in np.arange(0, xs.size):
        prob['turbineX'][0] = np.copy(xs[i])
        print(i)
        for j in np.arange(0, ys.size):
            if np.sqrt((ys[j]-boundary_center_y)**2 + (xs[i]-boundary_center_x)**2) > boundary_radius:
                vel_out[i, j] = np.nan
                continue
            prob['turbineY'][0] = np.copy(ys[j])
            prob.run_model()
            vel_out[i, j] = np.copy(prob['AEP'])
            # print(i*j+j)

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    # Hide grid lines
    ax.grid(False)
    plt.axis("off")

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    print(np.max(vel_out))
    ax.plot_surface(xx, yy, vel_out, vmin=np.nanmin(vel_out)-10, vmax=np.nanmax(vel_out)+10, cmap="hot", rstride=1, cstride=1, linewidth=0, antialiased=False)
    # ax.set_zlim(0, nTurbines*2E3)
    # plt.savefig('multimodal.pdf', transparent=True)

    plt.show()

    return 0

if __name__ == "__main__":

    # rank = mpi4py.MPI.COMM_WORLD.Get_rank()
    # size = mpi4py.MPI.COMM_WORLD.Get_size()
    #
    # wake_model = 1
    # opt_alg_number = 0
    # max_wec = 3
    #
    # wec_method_numbers = np.array([1, 2, 3])
    # nstepss = np.arange(1, 11)
    # layout_numbers = np.arange(0, 200)
    #
    # wmns, nss, lns = np.meshgrid(wec_method_numbers, nstepss, layout_numbers)
    # wmns = wmns.flatten()
    # nss = nss.flatten()
    # lns = lns.flatten()
    #
    # ntasks = wec_method_numbers.size*nstepss.size*layout_numbers.size
    #
    # for i in np.arange(0, ntasks):
    #
    #     if i % size != rank: continue
    #     print("Task number %d being done by processor %d of %d" % (i, rank, size))
    #     print(wmns[i], nss[i], lns[i])
    #     run_opt(lns[i], wmns[i], wake_model, opt_alg_number, max_wec, nss[i])

    # specify which starting layout should be used
    # layout_number = int(sys.argv[1])
    layout_number = 0
    # wec_method_number = int(sys.argv[2])
    wec_method_number = 0
    # model = int(sys.argv[3])
    model = 1
    # opt_alg_number = int(sys.argv[4])
    opt_alg_number = 0
    # max_wec = int(sys.argv[5])
    max_wec = 1
    # nsteps = int(sys.argv[6])
    nsteps = 1

    res = 3
    xs = np.linspace(0, 80.0*35, res)
    ys = np.linspace(0, 80.0 * 35, res)

    run_opt(layout_number, wec_method_number, model, opt_alg_number, max_wec, nsteps, xs, ys)
