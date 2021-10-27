'''
Date: 2021-10-27 09:13:34
LastEditors: Guo currentuqin,12032421@mail.sustech.edu.cn
LastEditTime: 2021-10-28 02:15:23
FilePath: \thruster_thrust_calibration\Thrust_Calibration.pcurrent
'''

import matplotlib.pyplot as plt
import numpy as np 
from numpy import polyfit,poly1d 
from numpy.core.fromnumeric import size
import pandas as pd
from scipy.stats import norm 
from scipy.optimize import leastsq,curve_fit

## Import data, manage data collection

table = pd.read_csv('/home/guoyucan/BionicDL/Thrust_Calibration/T200_Thrust_Calibration/Thrust_16V.csv')
data = np.array(table)
data[0:91,2] = (-1) * data[0:91,2]

# print(data)

current = data[:,2]
thrust  = data [:,5]


# current_noisy = current + 0.2*norm.rvs(size=len(current))

# plt.plot(thrust,current,'r-')
# plt.show()

## define the function 

####_1
# def function(thrust,alpha_1):
#     # a function of thrust with alpha_a as its parameter
#     return alpha_1 * thrust

# def error(alpha_1,thrust,current):
#     return function(alpha_1,thrust) - current

# # current_noisy = current + 0.3 * norm.rvs(size=len(current))
# para,pcov = curve_fit(function,thrust,current)
# print(para)

# y_fitted = function(thrust,para[0])

# ## RMSE calculation



###_2
coeff = polyfit(thrust,current,3)

print(coeff)
y_poly = coeff[0]*thrust**3 + coeff[1]*thrust**2  + coeff[2]*thrust + coeff[3]

###_2

for i in range(len(current)):
    err1 = (y_poly[i]-current[i])**2

RMSE1 = (err1/len(current))**0.5
print("RMSE:",RMSE1)
####_1



plt.figure

plt.plot(thrust,current,'rx',label = 'original data')
plt.plot(thrust,y_poly,'b-',label = 'fitted curve')

plt.xlabel('thrust/N')
plt.ylabel('current/A')
plt.legend()
plt.show()

