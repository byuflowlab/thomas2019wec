from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
# from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.utilities import sunflower_points
import time
import numpy as np
import matplotlib.pyplot as plt

#import cProfile
#import sys

if __name__ == "__main__":

    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    model = 1
    print(MODELS[model])
    wake_model_version = 2016

    sort_turbs = True
    wake_combination_method = 1 # can be [0:Linear freestreem superposition,
                                       #  1:Linear upstream velocity superposition,
                                       #  2:Sum of squares freestream superposition,
                                       #  3:Sum of squares upstream velocity superposition]
    ti_calculation_method = 0  # can be [0:No added TI calculations,
                                        #1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
                                        #2:TI by Niayifar and Porte Agel 2016,
                                        #3:TI by Niayifar and Porte Agel 2016 with soft max]

    if ti_calculation_method == 0:
        calc_k_star = False
    else:
        calc_k_star = True

    nRotorPoints = 1
    z_ref = 70.0
    z_0 = 0.0
    TI = 0.07

    # k_calc = 0.022
    k_calc = 0.3837*TI + 0.003678

    # define turbine size
    rotor_diameter = 80.  # (m)
    hub_height = 70.0

    z_ref = hub_height
    z_0 = 0.0

    # load performance characteristics
    cut_in_speed = 4.  # m/s
    rated_power = 2000.  # kW

    # air_density = 1.1716  # kg/m^3
    Ar = 0.25 * np.pi * rotor_diameter ** 2
    # cp_curve_vel = ct_curve[:, 0]
    air_density = 1.1716
    from scipy.interpolate import UnivariateSpline
    power_data = np.loadtxt('../project-code/input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
    # cp_curve_cp = niayifar_power_model(cp_curve_vel)/(0.5*air_density*cp_curve_vel**3*Ar)
    cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
    cp_curve_vel = power_data[:, 0]
    cp_curve_spline = UnivariateSpline(cp_curve_vel, cp_curve_cp, ext='const')
    cp_curve_spline.set_smoothing_factor(.0001)

    ct_curve = np.loadtxt('../project-code/input_files/mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    ct_curve_wind_speed = ct_curve[:, 0]
    ct_curve_ct = ct_curve[:, 1]
    # ct_curve = np.loadtxt('./input_files/predicted_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speed': ct_curve_wind_speed,
                          'interp_type': 1,
                          'differentiable': True}

    ######################### for MPI functionality #########################


    size = 1  # number of processors (and number of wind directions to run_model)

    #########################################################################
    # define turbine size
    rotor_diameter = 80.0 # (m)
    hub_height = 70.0

    turbineX = np.array([0.0, 3.*rotor_diameter, 7.*rotor_diameter, 10.*rotor_diameter])
    turbineY = np.array([-2.75*rotor_diameter, 2.75*rotor_diameter, 0.0, 0.0])

    # initialize input variable arrays
    nTurbs = turbineX.size
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
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.0  # deg.

    # Define flow properties
    windDirections = np.array([270.0])
    windFrequencies = np.ones(1)

    air_density = 1.1716  # kg/m^3
    shear_exp = 0.15
    sm_smoothing = 1000.

    wind_speed = 8.0  # m/s
    windSpeeds = np.ones(size) * wind_speed

    normalized_velocity = np.zeros(nTurbs)
    normalized_power = np.zeros(nTurbs)
    turbulence_intensity = np.zeros(nTurbs)

    if MODELS[model] == 'BPA':
        # initialize problem
        prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                              minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                              wake_model=gauss_wrapper, wake_model_options=wake_model_options,
                                              params_IdepVar_func=add_gauss_params_IndepVarComps,
                                              params_IdepVar_args={'nRotorPoints': nRotorPoints},
                                              cp_points=cp_curve_cp.size, cp_curve_spline=cp_curve_spline))
    elif MODELS[model] == 'FLORIS':
        # initialize problem
        prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                              minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                              wake_model=floris_wrapper,
                                              params_IdepVar_func=add_floris_params_IndepVarComps,
                                              params_IndepVar_args={}))
    elif MODELS[model] == 'JENSEN':
        # initialize problem
        prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                              minSpacing=minSpacing, differentiable=False, use_rotor_components=False,
                                              wake_model=jensen_wrapper,
                                              params_IdepVar_func=add_jensen_params_IndepVarComps,
                                              params_IndepVar_args={}))
    else:
        ValueError('The %s model is not currently available. Please select BPA or FLORIS' %(MODELS[model]))

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    print(('Problem setup took %.03f sec.' % (toc - tic)))

    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
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

    if MODELS[model] is 'BPA':
        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:calc_k_star'] = calc_k_star
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:wake_model_version'] = wake_model_version
        prob['model_params:I'] = TI
        prob['model_params:shear_exp'] = shear_exp
        prob['model_params:sm_smoothing'] = sm_smoothing
        if nRotorPoints > 1:
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

    exp_fac_values = np.arange(0.0, 30.1, 2.5)
    locations = np.arange(-6.*rotor_diameter, 6.*rotor_diameter, 0.5)
    powers0 = np.zeros([exp_fac_values.size, locations.size])
    powers1 = np.zeros([exp_fac_values.size, locations.size])
    powers2 = np.zeros([exp_fac_values.size, locations.size])
    powers3 = np.zeros([exp_fac_values.size, locations.size])
    aeps = np.zeros([exp_fac_values.size, locations.size])

    for k, i in zip(exp_fac_values, np.arange(0, exp_fac_values.size)):
        print('start %s run_model' % (MODELS[model]))
        prob['model_params:wec_spreading_angle'] = np.copy(k)
        print(k)
        for location, j in zip(locations, np.arange(0, locations.size)):
            # run_model the problem
            # tic = time.time()
            # cProfile.run_model('prob.run_model()')
            turbineY[2] = turbineY[3] = location
            prob['turbineY'] = turbineY
            prob.run_model()
            powers0[i, j] = prob['wtPower0'][0]
            powers1[i, j] = prob['wtPower0'][1]
            powers2[i, j] = prob['wtPower0'][2]
            powers3[i, j] = prob['wtPower0'][3]
            aeps[i, j] = prob['AEP']
            # print(powers)
            # print(aeps)
            # toc = time.time()
        # quit()

    for direction_id in range(0, windDirections.size):
        print('yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
        # print( 'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
    #     print( 'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

    print('turbine X positions in wind frame (m): %s' % prob['turbineX'])
    print('turbine Y positions in wind frame (m): %s' % prob['turbineY'])
    print('turbine hub wind velcities (m/s): %s' % prob['wtVelocity0'])
    print('wind farm power in each direction (kW): %s' % prob['dirPowers'])
    print('AEP (kWh): %s' % prob['AEP'])

    # TI_file = np.loadtxt("TIturbs_tmp.txt")
    # turbulence_intensity = TI_file

    # fig, ax = plt.subplots()
    # for x, y in zip(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter):
    #     circle_end = plt.Circle((x,y), 0.5, facecolor='none', edgecolor='k', linestyle='-', label='Turbines')
    #     ax.add_artist(circle_end)
    # # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    #
    # for i in range(0, nTurbs):
    #     ax.plot([turbineX[i] / rotor_diameter, prob['turbineX'][i] / rotor_diameter],
    #             [turbineY[i] / rotor_diameter, prob['turbineY'][i] / rotor_diameter], '--k')
    # plt.axis('equal')
    # # plt.show()

    fig, ax = plt.subplots(2,3)
    fig2, ax2 = plt.subplots(1)

    for k, i in zip(exp_fac_values, np.arange(0, exp_fac_values.size)):
        # print(powers[i, :])
        ax[0,0].plot(locations/rotor_diameter, powers0[i, :], label="std. dev. = %f*sigma" % k)
        ax[1,0].plot(locations/rotor_diameter, powers1[i, :], label="std. dev. = %f*sigma" % k)
        ax[0,1].plot(locations/rotor_diameter, powers2[i, :], label="std. dev. = %f*sigma" % k)
        ax[1,1].plot(locations/rotor_diameter, powers3[i, :], label="std. dev. = %f*sigma" % k)

        ax[0, 2].plot(locations/rotor_diameter, aeps[i, :], label="std. dev. = %.2f*sigma" % k)
        if np.mod(i,2) == 0:
            ax2.plot(locations/rotor_diameter, aeps[i, :], label="std. dev. = %.2f*sigma" % k)
        ax[1, 2].plot(locations/rotor_diameter, aeps[i, :], label="std. dev. = %.2f*sigma" % k)
    np.savetxt('smoothing_initial_test_ti.txt', np.c_[locations/rotor_diameter, aeps[0, :], aeps[2, :], aeps[4, :],
                                                   aeps[6, :], aeps[8, :], aeps[10, :], aeps[12, :]],
               header='location/diam, aep')

    ax[0,0].set_xlabel('Cross Stream Location')
    ax[0,0].set_ylabel('Downs Stream Turbine Power (kW)')
    ax[0,0].set_title('turbine 0')

    ax[1,0].set_xlabel('Cross Stream Location')
    ax[1,0].set_ylabel('Downs Stream Turbine Power (kW)')
    ax[1,0].set_title('turbine 1')

    ax[0,1].set_xlabel('Cross Stream Location')
    ax[0,1].set_ylabel('Downs Stream Turbine Power (kW)')
    ax[0,1].set_title('turbine 2')

    ax[1,1].set_xlabel('Cross Stream Location')
    ax[1,1].set_ylabel('Downs Stream Turbine Power (kW)')
    ax[1,1].set_title('turbine 3')

    ax[0, 2].set_xlabel('Cross Stream Location')
    ax[0, 2].set_ylabel('AEP (kWh)')
    ax[0, 2].set_title('AEP')

    ax2.set_xlabel('Cross Stream Location')
    ax2.set_ylabel('AEP, kWh')
    # ax2.set_title('AEP')

    # plt.legend(loc=1, ncol=1)
    plt.tick_params(right='off', top='off')
    # ax2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('wec-with-no-ti-angle.pdf', transparent=True)
    plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # print(turbineY, turbineX)
    # for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
    #     circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-')
    #     ax1.add_artist(circle_start)
    #     print(x, y)
    #
    #
    # # plt.axis('equal')
    # ax1.set_xlabel('Turbine X Position ($X/D_r$)')
    # ax1.set_ylabel('Turbine Y Position ($Y/D_r$)')
    # ax1.set_xlim([np.min(turbineX/rotor_diameter)-1., np.max(turbineX/rotor_diameter)+1.])
    # ax1.set_ylim([-np.max(turbineY/rotor_diameter)-1., np.max(turbineY/rotor_diameter)+1.])
    #
    # ax1.legend([circle_start], ['turbines'])
    # plt.show()