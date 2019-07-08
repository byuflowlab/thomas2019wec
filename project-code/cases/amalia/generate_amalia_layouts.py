import numpy as np
import matplotlib.pyplot as plt
from wakeexchange.GeneralWindFarmComponents import calculate_boundary, calculate_distance
from math import cos, sin

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle = np.radians(angle)
    ox, oy = origin
    px = point[:, 0]
    py = point[:, 1]

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)

    return qx, qy

def flip(x, y, axis):

    if axis == 'x':
        x_out = x
        y_out = -y + max(y)

    elif axis == 'y':
        x_out = -x + max(x)
        y_out = y

    return x_out, y_out

def amalia_farm_random_start(nTurbines, boundary_vertices, boundary_normals, rotor_diamter, min_spacing=1.0):

    widthX = np.max(boundary_vertices[:, 0])
    widthY = np.max(boundary_vertices[:, 1])

    min_spacing *= rotor_diameter

    locations = np.zeros([nTurbines, 2])
    # generate random points within the wind farm boundary
    count = 0
    for i in range(0, nTurbines):
        # print i
        good_point = False

        while not good_point:

            # generate random point in containing rectangle
            # print(np.random.rand(1, 2))
            locations[i] = np.random.rand(1, 2)*np.array([widthX, widthY])
            # print np.array([locations[i, :]])
            # print locations[i]
            distances = calculate_distance(np.array([locations[i, :]]), boundary_vertices, boundary_normals)
            # print distances[0]
            # determine if the point is inside the wind farm boundary
            good_point = True

            for j in np.arange(0, distances.size):
                if distances[0, j] < 0.0:
                    good_point = False

            n_bad_spacings = 0.0
            for turb in np.arange(0, nTurbines):
                if turb >= i:
                    continue
                spacing = np.sqrt((locations[turb, 0] - locations[i, 0]) ** 2 + (locations[turb, 1] - locations[i, 1]) ** 2)
                if spacing < min_spacing:
                    n_bad_spacings += 1

            if n_bad_spacings > 0:
                good_point = False

    # print calculate_distance(locations, boundary_vertices, boundary_normals)
    turbineX = locations[:, 0]
    turbineY = locations[:, 1]

    return turbineX, turbineY

def plot_farm(turbineX, turbineY, rotor_diameter, boundary_vertices,
                    save_start=False, show_start=False, save_file=None):

    if show_start:
        fig, ax = plt.subplots()
        for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
            circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
            ax.add_artist(circle_start)
        ax.plot(boundary_vertices[:, 0] / rotor_diameter, boundary_vertices[:, 1] / rotor_diameter)

        plt.axis('equal')
        # ax.legend([circle_start], ['turbines'])
        ax.set_xlabel('Turbine X Position ($X/D_r$)')
        ax.set_ylabel('Turbine Y Position ($Y/D_r$)')



    if save_start:
        if save_file is None:
            plt.savefig('amalia_farm.pdf')
        else:
            plt.savefig(save_file)

    plt.show()
def generate_amalia_layouts(nLayouts, nTurbines, rotor_diameter, base_spacing, output_directory=None, show=False):

    if nLayouts > 10 and show == True:
        raise ValueError("do you really want to see %i plots in serial?" % nLayouts)

    locations = np.loadtxt('layout_amalia.txt')

    x = locations[:, 0]
    y = locations[:, 1]
    center = [np.max(x)-np.min(x), np.max(y)-np.min(y)]
    x, y = rotate(center, locations, -90)
    x, y = flip(x, y, 'x')

    turbineX = x - np.min(x)
    turbineY = y

    for i in np.arange(0, turbineX.size):
        locations[i, 0] = turbineX[i]
        locations[i, 1] = turbineY[i]

    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    np.savetxt(output_directory+'nTurbs%i_spacing%i_layout_0.txt' % (nTurbines, base_spacing),
               np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
               header="turbineX, turbineY")

    plot_farm(turbineX, turbineY, rotor_diameter, boundaryVertices,
                     show_start=show)

    if nLayouts > 1:
        for L in np.arange(1, nLayouts):
            print L
            turbineX, turbineY = amalia_farm_random_start(turbineX.size, boundaryVertices, boundaryNormals,
                                                          rotor_diameter, min_spacing=1)
            # print turbineX, turbineY
            np.savetxt(output_directory+"nTurbs%i_spacing%i_layout_%i.txt" % (nTurbines, base_spacing, L),
                       np.c_[turbineX/rotor_diameter, turbineY/rotor_diameter],
                       header="turbineX, turbineY")

            plot_farm(turbineX, turbineY, rotor_diameter, boundaryVertices,
                      show_start=show)


if __name__ == "__main__":

    rotor_diameter = 80.

    nLayouts = 200
    nTurbines = 60
    base_spacing = 5.

    output_directory = "./layouts/"

    show_plots = False

    generate_amalia_layouts(nLayouts, nTurbines, rotor_diameter, base_spacing, output_directory, show=show_plots)
