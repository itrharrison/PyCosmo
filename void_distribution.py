"""
Python script for reproducing the
distribution of number density of voids
"""

import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import sqrt, log, exp, fabs, pi

#from hmf import *
from cosmology import *
#from survey import *
from powspec import PowSpec

def void_and_cloud(void_barrier, collapse_barrier):
  # Calculates D ; the void-and-cloud parameter
  return fabs(void_barrier) / (collapse_barrier + fabs(void_barrier))
  

def multiplicity_function(nu,D,void_barrier,collapse_barrier):
  """
  calculates equation (4) in Sheth & van de Weygaert
  approximating the infinite series in equation (1)
  """
  
  return (1/nu) * (nu/(2*pi))**0.5 * exp(-0.5*nu) \
  * exp((-1*fabs(void_barrier)*(D**2) \
  / (collapse_barrier*nu*4)) - (2*(D**4)/(nu**2)))
  

def scaled_void_distribution(nu,void_barrier=-2.81,collapse_barrier=1.06):
  """
  'A Hierarchy of Voids : Sheth & van de Weygaert'
  Reproduces a scaled distribution of void masses/sizes
  shown in figure(7)  
  """
  
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  return multiplicity_function(nu,D,void_barrier,collapse_barrier)
  

def void_radii_distribution(R,void_barrier=-2.81,\
collapse_barrier=1.06,cosm=Cosmology(),ps=PowSpec()):
  
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  # calculate volume from a given R
  V = (4 * pi * pow(R,3) * pow(1.7,3)) / 3
  
  # calculate mass of given volume element
  M = V * cosm.rho_m() / 1.7**3
  
  # get sigma from PowSpec class
  sigma = ps.sigma_wmap7fit(log(M))
  
  # calculate f(sigma)
  fSig = multiplicity_function(sigma,D,void_barrier,collapse_barrier)
  
  no_dens = (fSig) / (V) #(pow(1.7,6)*fV*9) / (4*math.pi*(R**4)*mean_density)
  
  return no_dens
  

if __name__ == '__main__':
  
  
  
  nu_range = np.arange(0.1,20,0.05)
  
  nod = []
  nod2 = []
  
  for R in nu_range:
    nod.append(void_radii_distribution(R,void_barrier=-2.7))
    nod2.append(void_radii_distribution(R,collapse_barrier=1.686,void_barrier=-2.7))
    
  plt.plot(nu_range,nod,nu_range,nod2)
  
  plt.yscale('log')
  plt.xscale('log')
  
  plt.xlabel(r'r [Mpc/h]', fontsize='20')
  plt.ylabel(r'dn/dlnr $(h/Mpc)^{3}$', fontsize='20')
  plt.legend((r'$\delta_{c}=1.06$',r'$\delta_{c}=1.69$'), prop={'size':20})
  
  
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
  plt.ylim(0.01,0.6)
  #plt.xlim(0.5,5)
  #plt.xscale('log')
  #plt.yscale('log')
  plt.xlabel(r'$\nu = (\delta_{V} / \sigma)^{2}$', fontsize='20')
  plt.ylabel(r'$f(\nu)$', fontsize='20')
  """
  
  plt.show()
  
