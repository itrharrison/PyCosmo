"""
(HOPEFULLY) USEFUL POWER SPECTRUM CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

from powspec_init import power_spectrum_P # CL power spectrum initialisation script
import cosmology
from numpy import sqrt, log, exp, fabs, pi, cos, sin
from scipy import integrate
import numpy as np

from matplotlib import pyplot as plt

class PowSpec:

  def __init__(self):
    """Constructor
    Default (and currently only available option) to a polynomial fit to
    WMAP7+BAO+H0 ML  parameter power spectrum from CAMB.
    
    FIXME: long-term goal is to hook this up to Eisenstein & Hu's
    analytic transfer function, a la cosmolopy.
    """
    
    self.sigma = self.sigma_wmap7fit
    self.dlnsigmadlnm = self.dlnsigmadlnm_wmap7fit
    self.label = "fitted to WMAP7->CAMB"
    self.clfile="wmap7_bao_h0_scalCls.dat"
    
  
  def display(self):
    """Display method to show power spectrum currently working with.
    """
    print("Power spectrum {}".format(self.label))
    
  
  def sigma_wmap7fit(self, lnm):
    """Root of matter variance smoothed with top hat window function on a scale
    specified by log(m)
    
    Polynomial fit to calculation from a CAMB power spectrum
    with WMAP7 parameters
    """
    return np.exp(18.0058 - 1.47523*lnm + 0.0385449*lnm*lnm - 0.0000112539*pow(lnm,4) + (1.3719e-9)*pow(lnm,6))
    
  
  def dlnsigmadlnm_wmap7fit(self, lnm):
    """Slope of root matter variance wrt log mass:
    d(log(sigma)) / d(log(m))
  
    Polynomial fit to calculation from a CAMB power spectrum 
    with WMAP7 parameters
    """
    return -1.47523 + 0.0770898*lnm - 0.0000450156*pow(lnm,3) + (8.23139e-9)*pow(lnm,5)
    
  
  def tophat_w(self, k, r):
    """
    Fourier transform of the real space tophat window function
    (eq.8 from A.Zentner 06)
    
    """
    
    return (3.*(sin(k*r) - k*r*cos(k*r)))/((k*r)**3.)
  
  def sigma_r_sq(self, r,z=0.0):
    """integrate the function in sigma_integral
    between the limits of k : 0 to inf. (log(k) : -20 to 20)"""
    
    return integrate.quad(self.sigma_integral,-20,20,args=(r,z),limit=10000)
    
  
  def sigma_integral(self, logk,r,z=0.0):
    """returns the integral required to calculate
       sigma squared (Coles & Lucchin pg.266) """
    
    k = exp(logk)
    
    return (k**3 / (2 * pi**2)) * self.tophat_w(k,r)**2 * power_spectrum_P(k)
    
  
  def sigma_r(self, r,z=0.0,cosm=cosmology.Cosmology()):
    """ returns sigma, at arbitrary z"""
    return sqrt(self.sigma_r_sq(r,z)) * cosm.growth(z)
    

if __name__ == "__main__":
  
  ps = PowSpec()
  
  
  sigmar = []
  
  rrange = np.arange(0,20,1)
  
  #for r in rrange:
    #sigmar.append(sigma_r(5))
  
  print ps.sigma_r(5)
  
"""
  plt.plot(rrange,sigmar)
  
  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')
  
  plt.show()
"""
  
  
  
