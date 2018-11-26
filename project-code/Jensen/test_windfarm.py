import unittest
import numpy as np
from JensenWindFarmTest2 import get_standard_direction

class DirectionConverterTest(unittest.TestCase):

    def test_GivenNorth_WhenConvertedToStandard_Returns270(self):

        # Create windrose direction that we know the answer to. Units in degrees.
        north_windrose_direction = 0.0

        # Create standard direction we know is equal to this direction. This is the direction the wind is GOING TO,
        # not where it's COMING FROM. Units in degrees.
        north_standard_direction = 270.0

        self.assertEqual(north_standard_direction, get_standard_direction(north_windrose_direction), msg='Incorrect standard direction.')

    def test_GivenEast_WhenConvertedToStandard_Returns180(self):

        # Create windrose direction that we know the answer to. Units in degrees.
        east_windrose_direction = 90.0

        # Create standard direction we know is equal to this direction. This is the direction the wind is GOING TO,
        # not where it's COMING FROM. Units in degrees.
        east_standard_direction = 180.0

        self.assertEqual(east_standard_direction, get_standard_direction(east_windrose_direction), msg='Incorrect standard direction.')

    def test_GivenSouth_WhenConvertedToStandard_Returns90(self):

        # Create windrose direction that we know the answer to. Units in degrees.
        south_windrose_direction = 180.0

        # Create standard direction we know is equal to this direction. This is the direction the wind is GOING TO,
        # not where it's COMING FROM. Units in degrees.
        south_standard_direction = 90.0

        self.assertEqual(south_standard_direction, get_standard_direction(south_windrose_direction), msg='Incorrect standard direction.')

    def test_GivenWest_WhenConvertedToStandard_Returns0(self):

        # Create windrose direction that we know the answer to. Units in degrees.
        west_windrose_direction = 270.0

        # Create standard direction we know is equal to this direction. This is the direction the wind is GOING TO,
        # not where it's COMING FROM. Units in degrees.
        west_standard_direction = 0.0

        self.assertEqual(west_standard_direction, get_standard_direction(west_windrose_direction), msg='Incorrect standard direction.')

class WindFarmRotationTest(unittest.TestCase):

    def test_GivenZeroRotation_WhenCheckingCoordinates_ReturnsOriginalCoordinates(self):

        pass

    def test_Given90DegreeRotation_WhenCheckingCoordinates_ReturnsINSERTSTUFFHERE(self):

        pass

if __name__ == '__main__':
    unittest.main()