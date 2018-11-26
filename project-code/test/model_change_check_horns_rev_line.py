from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from plantenergy.OptimizationGroups import OptAEP
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
# from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.utilities import sunflower_points, circumference_points, line_points
from _porteagel_fortran import ct_to_axial_ind_func

from _porteagel_fortran import ct_to_axial_ind_func
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

import cPickle as pickle

#import cProfile
#import sys


def niayifar_power_model(u):
    power_niayifar_model = 0.17819 * (u) ** 5. - 6.5198 * (u) ** 4. + \
                           90.623 * (u) ** 3. - 574.62 * (u) ** 2. + 1727.2 * (u) - 1975.
    return power_niayifar_model

if __name__ == "__main__":

    MODELS = ['FLORIS', 'BPA', 'JENSEN', 'LARSEN']
    model = 1
    wake_model_version = 2016
    print(MODELS[model])

    rotor_diameter = 80.0  # (m)

    sort_turbs = True
    wake_combination_method = 1 # can be [0:Linear freestreem superposition,
                                       #  1:Linear upstream velocity superposition,
                                       #  2:Sum of squares freestream superposition,
                                       #  3:Sum of squares upstream velocity superposition]

    sm_smoothing = 700.

    nRotorPoints = 100
    # location = 0.625 #0.5*1.25
    location = (0.7353288267358161 + 0.6398349246319044)/2.
    location = 0.69
    print(location)
    rotor_pnt_typ = 1      # can be [0: circumference points,
                            #         1: sunflower points
                            #         2: horizontal line at hub height

    fig, ax = plt.subplots()
    if nRotorPoints > 1:
        if rotor_pnt_typ == 0:
            x, y = circumference_points(nRotorPoints, location=location)

            # ax.set_xlim([-70, 70])
        elif rotor_pnt_typ == 1:
            x, y = sunflower_points(nRotorPoints)
            # ax.plot(x, y)
        elif rotor_pnt_typ == 2:
            x, y = line_points(nRotorPoints)
        else:
            [x, y] = [0.0, 0.0]
        print(x, y)
        # x, y = sunflower_points(nRotorPoints)

        x *= 0.5*rotor_diameter
        y *= 0.5*rotor_diameter
        ax.scatter(x, y)

    else:
        [x, y] = np.array([0.0, 0.0])
        plt.scatter(x, y)

    ax.axis("equal")
    plt.xlabel('Horizontal Distance From Hub (m)')
    plt.ylabel('Vertical Distance From Hub (m)')
    plt.show()

    # quit()
    z_ref = 70.0
    z_0 = 0.0002
    TI = 0.077
    # TI = 0.083
    shear_exp = 0.15

    hub_height = 70.0

    # k_calc = 0.022
    # k_calc = 0.03
    k_calc = 0.3837*TI + 0.003678

    ct_curve = np.loadtxt('../input_files/mfg_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    ct_curve_wind_speed = ct_curve[:, 0]
    ct_curve_ct = ct_curve[:, 1]
    # ct_curve = np.loadtxt('./input_files/predicted_ct_vestas_v80_niayifar2016.txt', delimiter=",")
    # filename = "../input_files/NREL5MWCPCT_dict.p"
    # # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    # import cPickle as pickle
    #
    # data = pickle.load(open(filename, "rb"))
    # ct_curve = np.zeros([data['wind_speed'].size, 2])
    # ct_curve[:, 0] = data['wind_speed']
    # ct_curve[:, 1] = data['CT']

    air_density = 1.1716  # kg/m^3
    Ar = 0.25 * np.pi * rotor_diameter ** 2
    cp_curve_vel = ct_curve[:, 0]
    power_data = np.loadtxt('../input_files/niayifar_vestas_v80_power_curve_observed.txt', delimiter=',')
    # cp_curve_cp = niayifar_power_model(cp_curve_vel)/(0.5*air_density*cp_curve_vel**3*Ar)
    cp_curve_cp = power_data[:, 1] * (1E6) / (0.5 * air_density * power_data[:, 0] ** 3 * Ar)
    cp_curve_wind_speed = power_data[:, 0]
    cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
    cp_curve_spline.set_smoothing_factor(.0001)
    # xs = np.linspace(0, 35, 1000)
    # plt.plot(xs, cp_curve_spline(xs))
    # plt.scatter(cp_curve_vel, cp_curve_cp)
    # plt.show()
    # quit()
    wake_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve_ct': ct_curve_ct,
                          'ct_curve_wind_speed': ct_curve_wind_speed,
                          'interp_type': 1}

    locations = np.loadtxt("../input_files/horns_rev_locations.txt", delimiter=",")
    turbineX = locations[:, 0] * rotor_diameter
    turbineY = locations[:, 1] * rotor_diameter
    nTurbines = turbineX.size

    nDirections = 1.

    # plt.plot(turbineX, turbineY, 'o')
    # plt.show()
    # quit()

    normalized_velocity = np.zeros([6, nTurbines])
    normalized_power = np.zeros([6, nTurbines])
    # turbulence_intensity = np.zeros([6, nTurbines])
    power_niayifar_model = np.zeros([6, nTurbines])

    normalized_power_averaged_NPA = np.zeros([6, 10])
    normalized_power_averaged_OURS = np.zeros([6, 10])
    normalized_velocity_averaged = np.zeros([6, 10])
    # turbulence_intensity_averaged = np.zeros([6, 10])

    for ti_calculation_method in np.array([0, 4, 5]):
                                        # can be [0:No added TI calculations,
                                        #1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
                                        #2:TI by Niayifar and Porte Agel 2016,
                                        #3:TI by Niayifar and Porte Agel 2016 with added soft max function]
        if ti_calculation_method == 0:
            calc_k_star = False
        else:
            calc_k_star = True

        ######################### for MPI functionality #########################
        from openmdao.core.mpi_wrap import MPI

        if MPI:  # pragma: no cover
            # if you called this script with 'mpirun', then use the petsc data passing
            from openmdao.core.petsc_impl import PetscImpl as impl

        else:
            # if you didn't use 'mpirun', then use the numpy data passing
            from openmdao.api import BasicImpl as impl


        def mpi_print(prob, *args):
            """ helper function to only print on rank 0 """
            if prob.root.comm.rank == 0:
                print(*args)


        prob = Problem(impl=impl)

        size = 1  # number of processors (and number of wind directions to run)

        #########################################################################
        # define turbine size


        # define turbine locations in global reference frame
        # original example case
        # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
        # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

        # # Scaling grid case
        # nRows = 3  # number of rows and columns in grid
        # spacing = 3.5  # turbine grid spacing in diameters
        #
        # # Set up position arrays
        # points = np.linspace(start=spacing * rotor_diameter, stop=nRows * spacing * rotor_diameter, num=nRows)
        # xpoints, ypoints = np.meshgrid(points, points)
        # turbineX = np.ndarray.flatten(xpoints)
        # turbineY = np.ndarray.flatten(ypoints)


        # turbineY[9] = rotor_diameter/2.0
        # np.savetxt('RoundFarm38Turb5DSpacing.txt', np.c_[turbineX+500.,turbineY+500.], header="TurbineX (m), TurbineY (m)")
        # locs = np.loadtxt('RoundFarm38Turb5DSpacing.txt')
        # x = locs[:, 0]/rotor_diameter
        # y = locs[:, 1]/rotor_diameter
        # plt.scatter(x, y)
        # set values for circular boundary constraint

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

        # print("N Turbines: ", nTurbs)

        # define initial values
        for turbI in range(0, nTurbs):
            rotorDiameter[turbI] = rotor_diameter  # m
            hubHeight[turbI] = hub_height  # m
            axialInduction[turbI] = 1.0 / 3.0
            Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
            # Ct[turbI] = 0.803
            # axialInduction[turbI] = ct_to_axial_ind_func(Ct[turbI])
            # print(Ct)
            # quit()
            # Cp[turbI] = (0.7737 / 0.944) * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
            Cp[turbI] = 4.0 * axialInduction[turbI] * np.power((1. - axialInduction[turbI]), 2)
            # Cp[turbI] = 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
            # generatorEfficiency[turbI] = 0.944
            generatorEfficiency[turbI] = 1.
            yaw[turbI] = 0.0  # deg.

        # Define flow properties
        windDirections = np.array([270.0]) #np.linspace(-15., 15., nDirections) + 270.0
        windFrequencies = np.ones(1)


        # print(windDirections)
        # print(windFrequencies)
        air_density = 1.1716  # kg/m^3

        wind_speed = 8.0  # m/s
        windSpeeds = np.ones(size) * wind_speed

        if MODELS[model] == 'BPA':
            # initialize problem
            prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, nVertices=0,
                                                  minSpacing=minSpacing, differentiable=True, use_rotor_components=False,
                                                  wake_model=gauss_wrapper, params_IdepVar_func=add_gauss_params_IndepVarComps,
                                                  params_IndepVar_args={'nRotorPoints': nRotorPoints}, wake_model_options=wake_model_options,
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
        mpi_print(prob, ('Problem setup took %.03f sec.' % (toc - tic)))

        # time.sleep(10)
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
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_wind_speed'] = cp_curve_wind_speed
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

        if MODELS[model] is 'BPA':
            # prob['generatorEfficiency'] = np.ones(nTurbines)
            prob['model_params:wake_combination_method'] = wake_combination_method
            prob['model_params:ti_calculation_method'] = ti_calculation_method
            prob['model_params:calc_k_star'] = calc_k_star
            prob['model_params:sort'] = sort_turbs
            prob['model_params:z_ref'] = z_ref
            prob['model_params:z_0'] = z_0
            prob['model_params:ky'] = k_calc
            prob['model_params:kz'] = k_calc
            prob['model_params:print_ti'] = True
            prob['model_params:wake_model_version'] = wake_model_version
            prob['model_params:sm_smoothing'] = sm_smoothing
            prob['model_params:shear_exp'] = shear_exp
            if nRotorPoints > 1:
                if rotor_pnt_typ == 0:
                    prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = circumference_points(nRotorPoints, location=location)
                if rotor_pnt_typ == 1:
                    prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)
                if rotor_pnt_typ == 2:
                    prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = line_points(nRotorPoints)
                # plt.scatter(prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'])
                # plt.axis('equal')
                # plt.show()
                # quit()
        # set options
        # prob['floris_params:FLORISoriginal'] = True
        # prob['floris_params:CPcorrected'] = False
        # prob['floris_params:CTcorrected'] = False

        # run the problem
        mpi_print(prob, 'start %s run' %(MODELS[model]))
        tic = time.time()
        # cProfile.run('prob.run()')
        prob.run()
        toc = time.time()

        for direction_id in range(0, windDirections.size):
            mpi_print(prob, 'yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
            # for direction_id in range(0, windDirections.size):
            # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
        # for direction_id in range(0, windDirections.size):
        #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

        mpi_print(prob, 'turbine X positions in wind frame (m): %s' % prob['turbineX'])
        mpi_print(prob, 'turbine Y positions in wind frame (m): %s' % prob['turbineY'])
        mpi_print(prob, 'turbine hub wind velcities (m/s): %s' % prob['wtVelocity0'])
        mpi_print(prob, 'wind farm power in each direction (kW): %s' % prob['dirPowers'])
        mpi_print(prob, 'AEP (kWh): %s' % prob['AEP'])

        for direction in np.arange(0, nDirections):
            normalized_velocity[ti_calculation_method, :] += prob['wtVelocity%i' % direction] / wind_speed
            print(prob['wtVelocity%i' % direction])
            normalized_power[ti_calculation_method, :] += prob['wtPower%i' % direction] / max(prob['wtPower%i' % direction])
        normalized_velocity[ti_calculation_method, :] /= nDirections
        normalized_power[ti_calculation_method, :] /= nDirections
        # quit()
        # power_niayifar_model[ti_calculation_method, :] = niayifar_power_model(prob['wtVelocity0'])/niayifar_power_model(wind_speed)
        power_niayifar_model[ti_calculation_method, :] = niayifar_power_model(prob['wtVelocity0'])/niayifar_power_model(prob['wtVelocity0'][0])


        # TI_file = np.loadtxt("TIturbs_tmp.txt")
        # turbulence_intensity[ti_calculation_method, :] = TI_file

        for row in np.arange(0, 10):
            normalized_power_averaged_OURS[ti_calculation_method, row] = np.average([normalized_power[ti_calculation_method, (10 * 0 + 40) + row],
                                                                                normalized_power[ti_calculation_method, (10 * 1 + 40) + row],
                                                                                normalized_power[ti_calculation_method, (10 * 2 + 40) + row]])
            normalized_power_averaged_NPA[ti_calculation_method, row] = np.average([power_niayifar_model[ti_calculation_method, (10 * 0 + 40) + row],
                                                                                power_niayifar_model[ti_calculation_method, (10 * 1 + 40) + row],
                                                                                power_niayifar_model[ti_calculation_method, (10 * 2 + 40) + row]])
            normalized_velocity_averaged[ti_calculation_method, row] = np.average([normalized_velocity[ti_calculation_method, (10 * 0 + 40) + row],
                                                                                   normalized_velocity[ti_calculation_method, (10 * 1 + 40) + row],
                                                                                   normalized_velocity[ti_calculation_method, (10 * 2 + 40) + row]])
            # turbulence_intensity_averaged[ti_calculation_method, row] = np.average([turbulence_intensity[ti_calculation_method, (10 * 0 + 40) + row],
            #                                                                         turbulence_intensity[ti_calculation_method, (10 * 1 + 40) + row],
            #                                                                         turbulence_intensity[ti_calculation_method, (10 * 2 + 40) + row]])


        power_niayifar_model[ti_calculation_method, :] = 0.17819 * (normalized_velocity[ti_calculation_method, :] * wind_speed) ** 5 - \
                               6.5198 * (normalized_velocity[ti_calculation_method, :] * wind_speed) ** 4 + \
                               90.623 * (normalized_velocity[ti_calculation_method, :] * wind_speed) ** 3 - \
                               574.62 * ( normalized_velocity[ti_calculation_method, :] * wind_speed) ** 2 + \
                               1727.2 * (normalized_velocity[ti_calculation_method, :] * wind_speed) - 1975
        power_niayifar_model[ti_calculation_method, :] /= power_niayifar_model[ti_calculation_method, 0]


        # turbulence_intensity[ti_calculation_method, :] = TI_file

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

    data_directory = "/Users/jaredthomas/Documents/projects/gaussian-yaw/project-code/Image_gen/image_data/"
    ti_data = np.loadtxt(data_directory+"niayifar-local-turbulence-intensity-les.txt", delimiter=",")
    ti_crespo_hernandez = np.loadtxt(data_directory+"niayifar-local-turbulence-intensity-crespo-hernandez.txt", delimiter=",")
    power_data_les = np.loadtxt(data_directory+"niayifar-normalized-power-les.txt", delimiter=",")
    power_data_model = np.loadtxt(data_directory+"niayifar-normalized-power-model.txt", delimiter=",")
    power_data_obs = np.loadtxt(data_directory+"beaucage2012_norm_power_hornz_rev_observations.txt", delimiter=",")
    velocity_data = np.loadtxt(data_directory+"van-leuven-normalized-velocity-measurements.txt", delimiter=",")
    velocity_data_npa = np.array([8., 5.5, 6., 6., 6., 6., 6., 6., 6., 6.])
    velocity_model_npa = np.array([8., 6.0, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5])

    power_curve_data = np.loadtxt(data_directory+"power_curve_v80.txt", delimiter=",")

    curve_power = np.interp(normalized_velocity_averaged[2, :]*wind_speed, power_curve_data[:, 0], power_curve_data[:, 1])
    # print(velocity_data[:, 0])
    # quit()
    # fig, ax = plt.subplots(3)
    #
    # # ax[1].scatter(np.round(velocity_data[:, 0]) - 1, velocity_data[:, 1], s=100, label='Van Leuven 1992', c='g', marker='s')
    #
    # ax[1].scatter(np.arange(0,10), velocity_data_npa/max(velocity_data_npa), label='NPA LES - guess', c='c', marker='s')
    # ax[1].scatter(np.arange(0,10), velocity_model_npa/max(velocity_model_npa), label='NPA Model - guess', c='g', marker='s')
    # ax[1].scatter(np.arange(0,10), normalized_velocity_averaged[0, :], label='No Local TI', c='b', marker='^')
    # # ax[1].scatter(np.arange(0,nTurbines), normalized_velocity[1, :], label='Jen TI', c='g', marker='s')
    # ax[1].scatter(np.arange(0,10), normalized_velocity_averaged[2, :], label='NPA TI', c='r', marker='*')
    # ax[1].scatter(np.arange(0,10), normalized_velocity_averaged[3, :], label='NPA TI w/SM', c='y', marker='>')
    #
    # ax[1].set_ylabel('Normalized Velocity ($U/ U_ \infty$)')
    # ax[1].set_xlabel('Turbine Number')
    # ax[1].set_ylim([0.0, 1.2])
    # ax[1].legend(loc=4, ncol=4)
    #
    #
    #
    # # TI_NPA = np.array([7.4999999999999997E-002, 0.18754638323820802, 0.17273243863345430, 0.17344141394463389,
    # #                0.17340405873765918, 0.17340601760246036, 0.17340591485608928, 0.17340592024527091,
    # #                0.17340591996260107, 0.17340591997742749])
    # # TI_NPA = np.array([7.4999999999999997E-002,  0.18074462275806025,       0.18509419334191782,
    # #                   0.18521380900661796,       0.18521706015365907,       0.18521714849144877,
    # #                   0.18521715089167778,       0.18521715095689445,       0.18521715095866648,
    # #                   0.18521715095871460])
    # # # TI_JA = np.array([7.4999999999999997E-002, 0.18754638323820802, 0.21075378195641487,  0.23254821047868099,
    # # #                  0.24962786733314535,  0.26371711478101056,  0.27574696635779561,
    # # #                  0.28627238764940771,  0.29564911808857880,  0.30411870272436770])
    # # TI_JA = np.array([7.4999999999999997E-002, 0.18074462275806025, 0.22718736948696139, 0.25778245014516005,
    # #                    0.28087507944316786, 0.29955032956224120, 0.31529905071415559,
    # #                    0.32895878629993602, 0.34104842481252906, 0.35191242053961508])
    # # TI_NONE = np.array([7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002,
    # # 7.4999999999999997E-002])
    #
    # ax[0].scatter(np.round(ti_data[:, 0])-1, ti_data[:, 1], label="Niayifar LES", c='c', marker='s')
    # ax[0].scatter(np.round(ti_data[:, 0])-1, ti_crespo_hernandez[:, 1], label="Niayifar C&H", c='c', marker='o')
    # ax[0].scatter(np.arange(0, 10), turbulence_intensity_averaged[0, :], label="No Local TI", c='b', marker='^')
    # # ax[0].scatter(np.arange(0, 10), turbulence_intensity_averaged[1, :], label="JA", c='g', marker='s' )
    # ax[0].scatter(np.arange(0, 10), turbulence_intensity_averaged[2, :], label="NPA TI", c='r', marker='*')
    # ax[0].scatter(np.arange(0, 10), turbulence_intensity_averaged[3, :], label="NPA TI w/SM", c='y', marker='>')
    # ax[0].scatter(np.arange(0, 10), turbulence_intensity_averaged[4, :], label="NPA TI w/AR", c='g', marker='o')
    # ax[0].set_xlabel('Turbine Number')
    # ax[0].set_ylabel('Turbulence Intensity')
    # ax[0].set_ylim([0.0, 0.45])
    # ax[0].legend(ncol=4)
    #
    # ax[2].scatter(np.round(power_data_les[:,0])-1, power_data_les[:, 1], label="Niayifar LES", c='c', marker='s')
    # ax[2].scatter(np.round(power_data_model[:,0])-1, power_data_model[:, 1], label="Niayifar Model", c='r', marker='s')
    # ax[2].scatter(2.0000000000000004-1, 0.46131468706600187, c='r', marker='*', edgecolor='r')
    # # ax[2].scatter(np.round(power_data_obs[:,0])-1, power_data_obs[:, 1], label="Beaucage 2012 obs", c='y', marker='s')
    # # ax[2].scatter(np.arange(0, 10), power_niayifar_model[0, :], label="power model no TI NPA", c='c', marker='o')
    # # ax[2].scatter(np.arange(0, 10), power_niayifar_model[2, :], label="power model with TI NPA", c='c', marker='o')
    # ax[2].scatter(np.arange(0, 10), normalized_power_averaged[0, :], label="No Local TI'", c='b', marker='^')
    # # ax[2].scatter(np.arange(0, nTurbines), normalized_power[1, :], label="JA", c='g', marker='s')
    # ax[2].scatter(np.arange(0, 10), normalized_power_averaged[2, :], label="NPA TI", c='r', marker='*')
    # ax[2].scatter(np.arange(0, 10), normalized_power_averaged[3, :], label="NPA TI w/SM", c='y', marker='>')
    # ax[2].scatter(np.arange(0, 10), normalized_power_averaged[4, :], label="NPA TI w/AR", c='g', marker='o')
    # ax[2].scatter(np.arange(0, 10), curve_power/curve_power[0], label="NPA TI using power curve", c='g', marker='+')
    #
    # print(curve_power/curve_power[0], power_curve_data)
    # ax[2].set_xlabel('Turbine Number')
    # ax[2].set_ylabel('Normalized Power ($P_i/P_0$)')
    # ax[2].set_ylim([0.0, 1.2])
    # ax[2].legend(ncol=4)

    #
    # ax[1].legend(loc=2)
    # plt.tight_layout()
    # plt.show()
    # plt.clf()

    fig, ax = plt.subplots(1)

    ax.plot(np.round(power_data_les[:, 0]) - 1, power_data_les[:, 1], label="Niayifar LES", c='r', marker='o')
    ax.plot(np.round(power_data_model[:, 0]) - 1, power_data_model[:, 1], label="Niayifar Model", c='b',
                  marker='v')
    print(normalized_power_averaged_NPA.shape)
    ax.plot(np.arange(0, 10), normalized_power_averaged_OURS[0, :], label="Ours w/our power - no loc TI", c='g',
            marker='o')
    # ax.plot(np.arange(0, 10), normalized_power_averaged_NPA[4, :], label="Our implementation w/NPA", c='k', marker='o')
    ax.plot(np.arange(0, 10), normalized_power_averaged_OURS[4, :], label="Ours w/our power", c='c', marker='o')
    # ax.plot(np.arange(0, 10), normalized_power_averaged_NPA[5, :], label="Our implementation w/NPA power and SM", c='k', marker='o', linestyle='--', fillstyle='none')
    ax.plot(np.arange(0, 10), normalized_power_averaged_OURS[5, :], label="Ours w/our power and SM", c='c', marker='o', linestyle='--', fillstyle='none')

    ax.set_xlabel('Turbine Row')
    ax.set_ylabel('Normalized Power ($P_i/P_0$)')
    ax.set_ylim([0.0, 1.2])
    ax.legend(ncol=1)

    plt.savefig("normalized_power_by_row.pdf", transparent=True)

    plt.show()

    # np.savetxt("power_line_res_%irotor pts_%imodel.txt" % (nRotorPoints, wake_model_version),
    #            np.c_[np.round(power_data_les[:, 0]) - 1,
    #                                        normalized_power_averaged_NPA[4, :],normalized_power_averaged_NPA[5, :],
    #                                        normalized_power_averaged_OURS[4, :], normalized_power_averaged_OURS[5, :]],
    #            header="turbine row, NPA power, NPA power w/SM, our power, our power w/SM (%i rotor pts, %i model)" % (nRotorPoints, wake_model_version))