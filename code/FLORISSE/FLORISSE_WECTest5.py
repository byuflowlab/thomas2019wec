"""
Filename: FLORISSE_WECTest3.py
Created by: Spencer McOmber
Created: Oct. 31, 2018
Description: Same purpose as FLORISSE_WECTest1.py, but now just looking for optimized wind turbine positions and
optimized AEP.
If this file doesn't have much to it, it's because I haven't gotten to fully writing it yet.
"""

prob.driver.add_desvar('turbineX', lower=np.zeros(nTurbines), upper=np.ones(nTurbines)*1000.0, scaler=1E-2)
prob.driver.add_desvar('turbineY', lower=np.ones(nTurbines)*min(turbineY), upper=np.ones(nTurbines)*max(turbineY), scaler=1E-2)

# add constraints
prob.driver.add_constraint('sc', lower=np.zeros(int(((nTurbines-1.)*nTurbines/2.))))

print('turbine x coordinates', prob['turbineX'])
print('turbine y coordinates', prob['turbineY'])
print('optimized AEP', prob['obj'])