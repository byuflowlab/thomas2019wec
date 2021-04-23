import numpy as np
from matplotlib import pyplot as plt
import openmdao.api as om
from plantenergy.OptimizationGroups import AEPGroup
from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps

def get_jensen_problem(modelvariant): 
    # define turbine size
    rotor_diameter = 40.0  # (m)
    hub_height = 90.0

    # define turbine locations in global reference frame
    turbineX = np.array([0.0, 6.0, 20.0])*rotor_diameter
    turbineY = np.array([0.0, 0.0, 0.0])

    z_ref = 90.0  # m
    z_0 = 0.0

    # load performance characteristics
    cut_in_speed = 3.  # m/s
    cut_out_speed = 25.  # m/s
    rated_wind_speed = 11.4  # m/s
    rated_power = 5000.  # kW
    generator_efficiency = 0.944

    input_directory = "../project-code/input_files/"
    filename = input_directory + "NREL5MWCPCT_dict.txt"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"

    data = np.loadtxt(filename)

    ct_curve = np.zeros([data[:, 0].size, 2])
    ct_curve_wind_speed = data[:, 0]
    ct_curve_ct = data[:, 2]

    # cp_curve_cp = data['CP']
    # cp_curve_wind_speed = data['wind_speed']

    loc0 = np.where(data[:, 0] < 11.55)
    loc1 = np.where(data[:, 0] > 11.7)

    cp_curve_cp = np.hstack([data[:, 1][loc0], data[:, 1][loc1]])
    cp_curve_wind_speed = np.hstack([data[:, 0][loc0], data[:, 0][loc1]])

    # initialize input variable arrays
    nTurbines = turbineX.size
    rotorDiameter = np.zeros(nTurbines)
    axialInduction = np.zeros(nTurbines)
    Ct = np.zeros(nTurbines)
    Cp = np.zeros(nTurbines)
    generatorEfficiency = np.zeros(nTurbines)
    yaw = np.zeros(nTurbines)

    # define initial values
    for turbI in range(0, nTurbines):
        rotorDiameter[turbI] = rotor_diameter  # m
        axialInduction[turbI] = 1.0 / 3.0
        Ct[turbI] = 4.0 * axialInduction[turbI] * (1.0 - axialInduction[turbI])
        Cp[turbI] = 0.7737 / 0.944 * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
        generatorEfficiency[turbI] = 1.0
        yaw[turbI] = 0.  # deg.

    # Define flow properties
    nDirections = 1
    wind_speed = 8.0 # m/s
    air_density = 1.1716  # kg/m^3
    wind_direction = 270.0  # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_frequency = 1.  # probability of wind in this direction at this speed

    # set up problem

    wake_model_options = {'nSamples': 0,
                            'nRotorPoints': 1,
                            'use_ct_curve': False,
                            'ct_curve_ct': ct_curve_ct,
                            'ct_curve_wind_speed': ct_curve_wind_speed,
                            'interp_type': 1,
                            'use_rotor_components': False,
                            'differentiable': False,
                            'verbose': False,
                            'variant': modelvariant}

    prob = om.Problem(model=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=jensen_wrapper,
                                        wake_model_options=wake_model_options,
                                        params_IdepVar_func=add_jensen_params_IndepVarComps,
                                        cp_points=cp_curve_cp.size,
                                        params_IdepVar_args={'use_angle': False}))

    # initialize problem
    prob.setup(check=False)

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
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

    prob['cp_curve_cp'] = cp_curve_cp
    prob['cp_curve_wind_speed'] = cp_curve_wind_speed
    # prob['model_params:spread_angle'] = 20.0
    # prob['model_params:alpha'] = 0.1

    # assign values to turbine states
    prob['cut_in_speed'] = np.ones(nTurbines) * cut_in_speed
    prob['cut_out_speed'] = np.ones(nTurbines) * cut_out_speed
    prob['rated_power'] = np.ones(nTurbines) * rated_power
    prob['rated_wind_speed'] = np.ones(nTurbines) * rated_wind_speed
    prob['use_power_curve_definition'] = True
    prob['gen_params:CTcorrected'] = True
    prob['gen_params:CPcorrected'] = True

    # run the problem
    prob.run_model()

    return prob

def plot_jensen(prob, ax, model, variant, res=100, rddown=6.0, rdcross=4.0):

    rotor_diam = prob["rotorDiameter"][0]

    y = np.linspace(-rdcross*rotor_diam, rdcross*rotor_diam, res)
    x = np.ones_like(y)*rotor_diam*rddown
    v = np.zeros_like(y)
    prob["turbineX"][1] = rotor_diam*rddown
    prob["rotorDiameter"][1] = 1.0
    for i in np.arange(0, len(y)):
        prob["turbineY"][1] = y[i]
        prob.run_model()
        v[i] = prob["wtVelocity0"][1]
    ax.plot(-v, y/rotor_diam)
    np.savetxt("model-profile-%s-%s.txt" %(model, variant), np.c_[x,y,v], header="x (m), y (m), v (m/s) for the %s %s model" %(model, variant))

if __name__ == "__main__":

    fig, ax = plt.subplots(2)
    variant = "TopHat"
    probth = get_jensen_problem(variant)
    plot_jensen(probth, ax[0], "jensen", variant, res=1000)

    variant = "CosineFortran"
    probcs = get_jensen_problem(variant)
    plot_jensen(probcs, ax[0], "jensen", variant, res=1000)

    plt.legend()
    plt.show()
