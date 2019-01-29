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

This file is being written to test the turbine wake models to see if WEC is being applied/used correctly.

A simple 4-turbine case will be considered to see if the wake models are being set up and used properly.
"""

