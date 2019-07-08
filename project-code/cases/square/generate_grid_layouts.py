import numpy as np
import matplotlib.pyplot as plt


def round_farm(rotor_diameter, center, radius, min_spacing=2.):
    # normalize inputs
    radius /= rotor_diameter
    center /= rotor_diameter

    # calculate how many circles can be fit in the wind farm area
    nCircles = np.floor(radius / min_spacing)
    radii = np.linspace(radius / nCircles, radius, nCircles)
    alpha_mins = 2. * np.arcsin(min_spacing / (2. * radii))
    nTurbines_circles = np.floor(2. * np.pi / alpha_mins)

    nTurbines = int(np.sum(nTurbines_circles)) + 1

    alphas = 2. * np.pi / nTurbines_circles

    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)

    index = 0
    turbineX[index] = center[0]
    turbineY[index] = center[1]
    index += 1
    for circle in np.arange(0, int(nCircles)):
        for turb in np.arange(0, int(nTurbines_circles[circle])):
            angle = alphas[circle] * turb
            w = radii[circle] * np.cos(angle)
            h = radii[circle] * np.sin(angle)
            x = center[0] + w
            y = center[1] + h
            turbineX[index] = x
            turbineY[index] = y
            index += 1

    return turbineX * rotor_diameter, turbineY * rotor_diameter


def round_farm_random_start(rotor_diameter, center, radius, min_spacing=2.):
    # normalize inputs
    radius /= rotor_diameter
    center /= rotor_diameter

    # calculate how many circles can be fit in the wind farm area
    nCircles = np.floor(radius / min_spacing)
    radii = np.linspace(radius / nCircles, radius, nCircles)
    alpha_mins = 2. * np.arcsin(min_spacing / (2. * radii))
    nTurbines_circles = np.floor(2. * np.pi / alpha_mins)

    nTurbines = int(np.sum(nTurbines_circles)) + 1

    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)

    # generate random points within the wind farm boundary
    for i in range(0, nTurbines):

        good_point = False

        while not good_point:

            # generate random point in containing rectangle
            print(np.random.rand(1, 2))
            [[turbineX[i], turbineY[i]]] = np.random.rand(1, 2) * 2. - 1.

            turbineX[i] *= radius
            turbineY[i] *= radius

            turbineX[i] += center[0]
            turbineY[i] += center[1]

            # calculate signed distance from the point to each boundary facet

            distance = radius - np.sqrt((turbineX[i] - center[0]) ** 2 + (turbineY[i] - center[1]) ** 2)

            # determine if the point is inside the wind farm boundary
            if distance > 0.0:
                good_point = True
                # sleep(0.05)
    return turbineX * rotor_diameter, turbineY * rotor_diameter


def square_farm_random_start(nTurbines, boundary_x, boundary_y, rotor_diamter, min_spacing=1.0):

    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)

    radius = rotor_diameter / 2.

    boundary_x[0] += radius
    boundary_y[0] += radius

    boundary_x[1] -= radius
    boundary_y[1] -= radius

    boundary_x /= rotor_diameter
    boundary_y /= rotor_diameter

    width = boundary_x[1] - boundary_x[0]

    center = [boundary_x[0] + width / 2., boundary_y[0] + width / 2.]

    # generate random points within the wind farm boundary
    count = 0
    for i in range(0, nTurbines):
        # print i
        good_point = False

        while not good_point:

            # generate random point in containing rectangle
            # print(np.random.rand(1, 2))
            [[turbineX[i], turbineY[i]]] = np.random.rand(1, 2) * width

            turbineX[i] += boundary_x[0]
            turbineY[i] += boundary_y[0]

            # determine if the point is inside the wind farm boundary
            if turbineX[i] >= boundary_x[0] and turbineX[i] <= boundary_x[1] and \
                    turbineY[i] >= boundary_y[0] and turbineY[i] <= boundary_y[1]:
                n_bad_spacings = 0
                for turb in np.arange(0, nTurbines):
                    if turb >= i:
                        continue
                    spacing = np.sqrt((turbineX[turb]-turbineX[i])**2+(turbineY[turb]-turbineY[i])**2)
                    if spacing < min_spacing:
                        n_bad_spacings += 1
                if n_bad_spacings == 0:
                    good_point = True



    return turbineX * rotor_diameter, turbineY * rotor_diameter


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

def plot_square_farm(turbineX, turbineY, rotor_diameter, boundary_x, boundary_y, boundary_width,
                    save_start=False, show_start=False, save_file=None):

    full_bound_x = np.array([boundary_x[0], boundary_x[0], boundary_x[1], boundary_x[1], boundary_x[0]])
    full_bound_y = np.array([boundary_y[0], boundary_y[1], boundary_y[1], boundary_y[0], boundary_y[0]])

    real_bound_x = np.array(
        [boundary_x[0] + rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2., boundary_x[1] - rotor_diameter / 2.,
         boundary_x[1] - rotor_diameter / 2., boundary_x[0] + rotor_diameter / 2.])
    real_bound_y = np.array(
        [boundary_y[0] + rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2., boundary_y[1] - rotor_diameter / 2.,
         boundary_y[0] + rotor_diameter / 2., boundary_y[0] + rotor_diameter / 2.])

    if show_start:
        plt.show()
        fig, ax = plt.subplots()
        for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
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


def generate_grid_layouts(nLayouts, nTurbines, rotor_diameter, base_spacing, output_directory=None, show=False):

    if nLayouts > 10 and show == True:
        raise ValueError("do you really want to see %i plots in serial?" % nLayouts)

    boundary_x = np.array([0.0, base_spacing * rotor_diameter * (np.sqrt(nTurbines)-1) + rotor_diameter])
    boundary_y = np.array([0.0, base_spacing * rotor_diameter * (np.sqrt(nTurbines)-1) + rotor_diameter])
    x = np.linspace(rotor_diameter / 2., base_spacing * rotor_diameter * (np.sqrt(nTurbines)-1) + rotor_diameter / 2., np.sqrt(nTurbines))
    y = np.linspace(rotor_diameter / 2., base_spacing * rotor_diameter * (np.sqrt(nTurbines)-1) + rotor_diameter / 2., np.sqrt(nTurbines))
    [xx, yy] = np.meshgrid(x, y)

    turbineX = xx.flatten()
    turbineY = yy.flatten()

    np.savetxt(output_directory+'nTurbs%i_spacing%i_layout_0.txt' % (nTurbines, base_spacing),
               np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
               header="turbineX, turbineY")

    plot_square_farm(turbineX, turbineY, rotor_diameter, boundary_x, boundary_y, boundary_x[1] - boundary_x[0],
                     show_start=show)

    if nLayouts > 1:
        for L in np.arange(1, nLayouts):
            turbineX, turbineY = square_farm_random_start(nTurbines, np.copy(boundary_x), np.copy(boundary_y), rotor_diameter)

            np.savetxt(output_directory+"nTurbs%i_spacing%i_layout_%i.txt" % (nTurbines, base_spacing, L),
                       np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
                       header="turbineX, turbineY")

            plot_square_farm(turbineX, turbineY, rotor_diameter, boundary_x, boundary_y, boundary_x[1] - boundary_x[0],
                             show_start=show)

    if show_plots:
        plt.show()

if __name__ == "__main__":

    rotor_diameter = 126.4

    nLayouts = 200
    nTurbines = 9
    base_spacing = 5.

    output_directory = "./layouts/"

    show_plots = False

    generate_grid_layouts(nLayouts, nTurbines, rotor_diameter, base_spacing, output_directory, show=show_plots)
