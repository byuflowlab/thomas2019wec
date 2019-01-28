import numpy as np
from matplotlib import pylab as plt
from time import sleep

def round_farm(rotor_diameter, center, radius, min_spacing=2.):

    # normalize inputs
    radius /= rotor_diameter
    center /= rotor_diameter

    # calculate how many circles can be fit in the wind farm area
    nCircles = np.floor(radius/min_spacing)
    radii = np.linspace(radius/nCircles, radius, nCircles)
    alpha_mins = 2.*np.arcsin(min_spacing/(2.*radii))
    nTurbines_circles = np.floor(2. * np.pi / alpha_mins)

    nTurbines = int(np.sum(nTurbines_circles))+1

    alphas = 2.*np.pi/nTurbines_circles

    turbineX = np.zeros(nTurbines)
    turbineY = np.zeros(nTurbines)

    index = 0
    turbineX[index] = center[0]
    turbineY[index] = center[1]
    index += 1
    for circle in np.arange(0, int(nCircles)):
        for turb in np.arange(0, int(nTurbines_circles[circle])):
            angle = alphas[circle]*turb
            w = radii[circle]*np.cos(angle)
            h = radii[circle]*np.sin(angle)
            x = center[0] + w
            y = center[1] + h
            turbineX[index] = x
            turbineY[index] = y
            index += 1

    return turbineX*rotor_diameter, turbineY*rotor_diameter


def plot_round_farm(turbineX, turbineY, rotor_diameter, boundary_center, boundary_radius, min_spacing=2.,
                    save_start=False, show_start=False, save_file=None):


    boundary_center_x = boundary_center[0]
    boundary_center_y = boundary_center[1]

    boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')
    farm_boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter+0.5, facecolor='none', edgecolor='k', linestyle='-')

    fig, ax = plt.subplots()
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    ax.add_patch(boundary_circle)
    ax.add_patch(farm_boundary_circle)
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


def round_farm_random_start(rotor_diameter, center, radius, min_spacing=2., min_spacing_random=1.):
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
            # print(np.random.rand(1, 2))
            [[turbineX[i], turbineY[i]]] = np.random.rand(1,2)*2. -1.

            turbineX[i] *= radius
            turbineY[i] *= radius

            turbineX[i] += center[0]
            turbineY[i] += center[1]

            # calculate signed distance from the point to each boundary facet

            distance = radius - np.sqrt((turbineX[i]-center[0])**2+(turbineY[i]-center[1])**2)

            # determine if the point is inside the wind farm boundary
            if distance > 0.0:

                n_bad_spacings = 0
                for turb in np.arange(0, nTurbines):
                    if turb >= i:
                        continue
                    spacing = np.sqrt((turbineX[turb] - turbineX[i]) ** 2 + (turbineY[turb] - turbineY[i]) ** 2)
                    if spacing < min_spacing_random:
                        n_bad_spacings += 1
                if n_bad_spacings == 0:
                    good_point = True
                # sleep(0.05)

        # print i
    return turbineX*rotor_diameter, turbineY*rotor_diameter

def generate_round_layouts(nLayouts, rotor_diameter, farm_center=0., farm_radius=4000., base_spacing=5., min_spacing=1.,
                           output_directory=None, show=False, save_layouts=False):

    if nLayouts > 10 and show == True:
        raise ValueError("do you really want to see %i plots in serial?" % nLayouts)

    turbineX, turbineY = round_farm(np.copy(rotor_diameter), np.copy(farm_center), np.copy(farm_radius), np.copy(base_spacing))

    nTurbines = turbineY.size

    print nTurbines

    area = np.pi * boundary_radius ** 2
    print area
    effective_rows = np.sqrt(38)
    print effective_rows
    effective_row_length = np.sqrt(area)
    print effective_row_length
    effective_spacing = effective_row_length / (effective_rows - 1.)
    print effective_spacing / rotor_diameter
    # plot_round_farm(turbineX, turbineY, rotor_diameter, center, boundary_radius, start_min_spacing, show_start=True)

    if save_layouts:
        np.savetxt(output_directory+'nTurbs%i_spacing%i_layout_0.txt' % (nTurbines, base_spacing),
                   np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
                   header="turbineX, turbineY")

    plot_round_farm(np.copy(turbineX), np.copy(turbineY), np.copy(rotor_diameter), np.copy(farm_center),
                    np.copy(farm_radius), show_start=show)


    if nLayouts > 1:
        for L in np.arange(1, nLayouts):
            print "Generating Layout %i" %L
            turbineX, turbineY = round_farm_random_start(np.copy(rotor_diameter), np.copy(farm_center),
                                                         np.copy(farm_radius), np.copy(base_spacing),
                                                         min_spacing_random=min_spacing)

            if save_layouts:
                np.savetxt(output_directory+"nTurbs%i_spacing%i_layout_%i.txt" % (nTurbines, base_spacing, L),
                           np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
                           header="turbineX, turbineY")

            plot_round_farm(np.copy(turbineX), np.copy(turbineY), np.copy(rotor_diameter), np.copy(farm_center),
                            np.copy(farm_radius), show_start=show)

if __name__ == "__main__":

    rotor_diameter = 126.4

    # boundary_radius = 0.5 * (4000 - rotor_diameter)  # 1936.8
    boundary_radius = 900.0
    center = np.array([boundary_radius, boundary_radius]) # looks ok, but Jared said (0, 0)...
    start_min_spacing = 5.

    nLayouts = 200

    show_layouts = False
    save_layouts = True

    output_directory = './input_files/layouts/round_9turbs/'

    # turbineX, turbineY = round_farm(np.copy(rotor_diameter), np.copy(center), np.copy(boundary_radius),
    #                                 min_spacing=start_min_spacing)

    generate_round_layouts(nLayouts, rotor_diameter, center, boundary_radius,
                           start_min_spacing, min_spacing=1.,
                           output_directory=output_directory, show=show_layouts, save_layouts=save_layouts)



