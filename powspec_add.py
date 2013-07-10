"""
'powspec_add.py'

Power spcetrum utilities script

calculates various parameters of the cosmological power spectrum

"""

from cosmology import * # IH cosmology class
from powspec_init import * # CL power spectrum script

from numpy import sqrt, log, exp, fabs, pi, sin, cos
from scipy import integrate
import numpy

from matplotlib import pyplot as plt

def tophat_w(k, r):
  """
  Fourier transform of the real space tophat window function
  (eq.8 from A.Zentner 06)
  
  """
  
  return (3.*(sin(k*r) - k*r*cos(k*r)))/((k*r)**3.)

def sigma_r_sq(r,z=0.0,cosm=Cosmology()):
  """integrate the function in sigma_integral
  between the limits of k : 0 to inf. (log(k) : -20 to 20)"""
  
  return integrate.quad(sigma_integral,-20,20,args=(r,z),limit=10000)
  

def sigma_integral(logk,r,z=0.0,cosm=Cosmology()):
  """returns the integral required to calculate
     sigma squared (Coles & Lucchin pg.266) """
  
  k = exp(logk)
  
  return (k**3 / (2 * pi**2)) * tophat_w(k,r)**2 * power_spectrum_P(k)
  

def sigma_r(r,z=0.0):
  """ returns sigma, at arbitrary z"""
  return sqrt(sigma_r_sq(r,z)) * cosm.growth(z)
  

if __name__ == "__main__":
  
  sigmar = []
  
  rrange = numpy.arange(0,20,1)
  
  #for r in rrange:
    #sigmar.append(sigma_r(5))
  
  print sigma_r(5)
  
"""
  plt.plot(rrange,sigmar)
  
  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')
  
  plt.show()
"""
  
  
  
  
  
  
  
  






