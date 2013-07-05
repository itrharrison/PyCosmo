"""
Python script for reproducing the distribution of
number density of voids in Sheth & van de Weygaert 04
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import sqrt, log, exp, fabs, pi

#from hmf import *
#from cosmology import *
#from survey import *

def scaled_void_distribution(nu,void_barrier=-2.81,collapse_barrier=1.06):
  
  # D ; the void-and-cloud parameter
  D = math.fabs(void_barrier) / (collapse_barrier + math.fabs(void_barrier))
  
  #print D
  #print nu
  
  return (1/nu) * (nu/(2*math.pi))**0.5 * exp(-0.5*nu) \
  * exp((-1*math.fabs(void_barrier)*(D**2) \
  / (collapse_barrier*nu*4)) - (2*(D**4)/(nu**2)))
  

def void_radii_distribution(R,void_barrier=-2.81,collapse_barrier=1.06):
  
  """
  Universe average density (from Peacock)
  1.8791 * pow(10,-26) Omega h^2 kg m^-3
  2.7755 * pow(10,11) Omega h^2 M(sol) Mpc^-3  
  """
  rho = 2.7755 * pow(10,11)#1.8791 * pow(10,-26)
  
  # D ; the void-and-cloud parameter
  D = math.fabs(void_barrier) / (collapse_barrier + math.fabs(void_barrier))
  
  V = (4 * math.pi * pow(R,3) * pow(1.7,3)) / 3
  
  #print V
  
  fV = (1/V) * (V/(2*math.pi))**0.5 * exp(-0.5*V) \
  * exp((-1*math.fabs(void_barrier)*(D**2) \
  / (collapse_barrier*V*4)) - ((2*(D**4))/(V**2)))
  
  #print fV
  
  no_dens = (pow(1.7,6)*fV*9) / (4*math.pi*(R**4)*rho)
  
  return no_dens
  

if __name__ == '__main__':
  nu_range = np.arange(0.1,5,0.1)
  
  nod = []
  
  for R in nu_range:
    nod.append(void_radii_distribution(R))
    
  plt.plot(nu_range,nod)
  
  """
  fnu1 = []
  fnu2 = []
  fnu3 = []
  
  for nu in nu_range:
    fnu1.append(scaled_void_distribution(nu,collapse_barrier=1.06))
    fnu2.append(scaled_void_distribution(nu,collapse_barrier=1.69))
    fnu3.append(scaled_void_distribution(nu,collapse_barrier=99999999))
  
  plt.plot(nu_range,fnu1,nu_range,fnu2,nu_range,fnu3)
  plt.legend((r'$\delta_{c}=1.06$',r'$\delta_{c}=1.69$',r'$\delta_{c}=\infty$'), prop={'size':20})
  plt.ylim(0,0.6)
  plt.xlabel(r'$\nu = (\delta_{V} / \sigma)^{2}$', fontsize='20')
  plt.ylabel(r'$f(\nu)$', fontsize='20')
  """
  
  plt.show()
  
