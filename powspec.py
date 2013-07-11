"""
(HOPEFULLY) USEFUL POWER SPECTRUM CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

#import powspec_init # CL power spectrum initialisation script

from numpy import sqrt, log, exp, fabs, pi, cos, sin
from scipy import integrate
from scipy import interpolate
import numpy as np
import sys

import EH.power as power
import constants as ct

from matplotlib import pyplot as plt

class PowSpec:

  def __init__(self,Dz=0.76226768,h0=0.702e0,om=0.274e0,ode=0.725e0,ob=0.0458):
    """Constructor
    Default (and currently only available option) to a polynomial fit to
    WMAP7+BAO+H0 ML  parameter power spectrum from CAMB.
    
    FIXME: long-term goal is to hook this up to Eisenstein & Hu's
    analytic transfer function, a la cosmolopy.
    """
    
    self.h_0 = h0
    self.O_m0 = om
    self.O_de0 = ode
    self.O_b0 = ob
    self.Dz = Dz
    
    self.sigma = self.sigma_wmap7fit
    self.dlnsigmadlnm = self.dlnsigmadlnm_wmap7fit
    self.label = "fitted to WMAP7->CAMB"
    self.clfile="wmap7_bao_h0_scalCls.dat"
    
  
  def display(self):
    """Display method to show power spectrum currently working with.
    """
    print("Power spectrum {}".format(self.label))
    
  
  def import_powerspectrum(self,ident,z=0.0):
    """import a transfer function from a CAMB produced output file"""
    
    z=str(int(z))
    
    k_array = []
    P_array = []
    
    for line in open('camb_{0}_matterpower_z{1}.dat'.format(ident,z),'r'):
      a, b = line.split()
      k_array.append(float(a))
      P_array.append(float(b))
    
    return k_array, P_array
    
  def interpolate_camb(self,k_array,P_array):
    """ returns a function that uses interpolation to find 
        the value of new points """
    
    return interpolate.interp1d(k_array,P_array)
    
  
  def transfer_function_EH(self,k,z=0.0):
    """Calculates transfer function given wavenumber"""
    
    # set cosmology
    power.TFmdm_set_cosm(self.O_m0,self.O_b0,-1,0,self.O_de0,self.h_0,z)
    
    """Call to EH power.c script
       h Mpc^-1 OR Mpc^-1 ???? """
    #return power.TFmdm_onek_mpc(k)
    return power.TFmdm_onek_mpc(k)
    
  
  def power_spectrum_P(self,k,z=0.0):
    """ returns the power spectrum P(k)"""
    
    n = 0.966
    
    delta_h = 1.94 * (10**-5) * self.O_m0**(-0.785 - (0.05*log(self.O_m0))) * exp(-0.95*(n-1)-0.169*(n-1)**2)
    
    Tk = self.transfer_function_EH(k)
    
    c_l = ct.const["c"] / ct.convert["Mpc_m"]   # speed of light in Mpc s^-1
    
    return (delta_h**2 * 2. * pi**2. * k**n) * (c_l/(self.h_0 * ct.convert['H0']))**(3.+n) * (Tk*self.Dz)**2
    
  
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
    
  
  def sigma_integral(self,logk,r,z=0.0):
    """returns the integral required to calculate
       sigma squared (Coles & Lucchin pg.266) """
    
    k = exp(logk)
    
    return (k**3 / (2 * pi**2)) * self.tophat_w(k,r)**2 * self.power_spectrum_P(k)
    
  
  def sigma_r(self, r,z=0.0):
    """ returns sigma, for radius r at arbitrary z"""
    return sqrt(self.sigma_r_sq(r,z)) * self.Dz
    
  
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

if __name__ == "__main__":
  
  ps = PowSpec()
  
  Pk = []
  Pnew = []
  knew = []
  
  k_array, P_array = ps.import_powerspectrum(ident="03269385")
  
  # find limits of camb power spectrum
  maxk = max(k_array)
  mink = min(k_array)
  
  # return interpolation function
  fint = ps.interpolate_camb(k_array,P_array)
  
  krange = np.logspace(-4,2,1000)
  
  for k in krange:
    Pk.append(ps.power_spectrum_P(k))
    if k > mink and k < maxk:
      Pnew.append(fint(k))
      knew.append(k)
  
  #plt.plot(krange,Pk)
  #plt.plot(k_array,P_array,knew,Pnew,krange,Pk)
  
  plt.yscale('linear')
  plt.xscale('linear')
  #plt.yscale('log')
  #plt.xscale('log')
  
  #plt.show()
  
  sigmar = []
  
  rrange = np.arange(0,20,1)
  
  for r in rrange:
    sigmar.append(ps.sigma_r(r))
  
  plt.plot(rrange,sigmar)
  
  plt.show()
  """
  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')
  
  plt.show()
"""
  
  
  
