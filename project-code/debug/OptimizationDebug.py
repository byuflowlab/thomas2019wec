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

    # Define initial turbine coordinates.
    turbineX = []