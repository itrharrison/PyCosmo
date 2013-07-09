"""
'powspec_add.py'

Power spcetrum utilities script

calculates various parameters of the cosmological power spectrum

"""

# Cosmolopy package
from cosmolopy.EH import power as power
from cosmolopy import perturbation as pert
from cosmolopy import constants as const

from cosmology import * # IH cosmology class
from powspec_init import * # CL power spectrum script

from numpy import sqrt, log, exp, fabs, pi
from scipy import integrate
import numpy

from matplotlib import pyplot as plt


def sigma_r_sq(r,z=0.0,cosm=Cosmology()):
  """integrate the function in sigma_integral
  between the limits of k : 0 to inf. (log(k) : -20 to 20)"""
  
  return integrate.quad(sigma_integral,-20,20,args=(r,z),limit=10000)
  

def sigma_integral(logk,r,z=0.0,cosm=Cosmology()):
  """returns the integral required to calculate
     sigma squared (Coles & Lucchin pg.266) """
  
  k = exp(logk)
  
  return (k**3 / (2 * pi**2)) * pert.w_tophat(k,r)**2 * power_spectrum_P(k)
  

def sigma_r(r,z=0.0):
  """ returns sigma, at arbitrary z"""
  return numpy.sqrt(sigma_r_sq(r,z)) * growth_factor_D(z)
  

if __name__ == "__main__":
  
  
  
  
