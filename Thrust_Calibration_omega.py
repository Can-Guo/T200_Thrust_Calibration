'''
*********************************************************************************************
  *File: Thrust_Calibration_omega.py
  *Project: T200_Thrust_Calibration
  *Filepath: /home/guoyucan/BionicDL/Thrust_Calibration/T200_Thrust_Calibration/Thrust_Calibration_omega.py 
  *File Created: Thursday, 28th October 2021 4:02:28 am
  *Author: Guo Yucan, 12032421@mail.sustech.edu.cn 
  *Last Modified: Thursday, 28th October 2021 4:02:33 am
  *Modified By: Guo Yucan, 12032421@mail.sustech.edu.cn 
  *Copyright @ 2021 , BionicDL LAB, SUSTECH, Shenzhen, China 
*********************************************************************************************
'''
import matplotlib.pyplot as plt
import numpy as np 
from numpy import polyfit,poly1d 
from numpy.core.fromnumeric import size
import pandas as pd
from scipy.stats import norm 
from scipy.optimize import leastsq,curve_fit

## Import data, manage data collection

table = pd.read_csv('/home/guoyucan/BionicDL/Thrust_Calibration/T200_Thrust_Calibration/Thrust_12V.csv')
data = np.array(table)
data[0:91,1] = (-1) * data[0:91,1]

# print(data)

omega_rpm = data[:,1]
omega = omega_rpm * np.pi / 30.0
thrust  = data [:,5]


## define the function 

####_1
# def function(thrust,alpha_1):
#     # a function of thrust with alpha_a as its parameter
#     return alpha_1 * thrust * thrust

# def error(alpha_1,thrust,omega):
#     return function(alpha_1,thrust) - omega

# # omega_noisy = omega + 0.3 * norm.rvs(size=len(omega))
# para,pcov = curve_fit(function,thrust,omega)
# print(para)

# y_fitted = function(thrust,para[0])

# ## RMSE calculation


##_2
coeff = polyfit(thrust,omega,3)

print(coeff)
y_poly = coeff[0]*thrust**3 + coeff[1]*thrust**2  + coeff[2]*thrust + coeff[3]

###_2

for i in range(len(omega)):
    err1 = (y_poly[i]-omega[i])**2

RMSE1 = (err1/len(omega))**0.5
print("RMSE:",RMSE1)
####_1



plt.figure

plt.plot(thrust,omega,'r.',label = 'original data')
plt.plot(thrust,y_poly,'b-',label = 'fitted curve')

plt.xlabel('thrust/N')
plt.ylabel('omega/A')
plt.legend()
plt.show()