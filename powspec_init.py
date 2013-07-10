"""
'powspec_init.py'

Generic Power spectrum calculator

produces a usable power spectrum from CAMB files or the Eisenstein & Hu analytic function

"""

# Cosmolopy package
from cosmolopy.EH import power as power
from cosmolopy import perturbation as pert
from cosmolopy import constants as const

from cosmology import * # IH cosmology class

from numpy import sqrt, log, exp, fabs, pi
from scipy import integrate
import numpy
import itertools

from matplotlib import pyplot as plt


def import_transfer_function(z=0.0):
  """import a transfer function from a CAMB produced output file"""
  
  k_array = []
  T_array = []
  
  for line in open('camb_03269385_matterpower_z%s.dat' % int(z),'r'):
    a, b = line.split()
    k_array.append(float(a))
    T_array.append(float(b))
  
  return k_array, T_array

def transfer_function_EH(k,z=0.0,cosm=Cosmology()):
  """Calculates transfer function given wavenumber"""
  
  # set cosmology
  power.TFmdm_set_cosm(cosm.O_m0,cosm.O_b0,-1,0,cosm.O_de0,cosm.h_0,z)
  
  """Call to EH power.c script
     h Mpc^-1 OR Mpc^-1 ???? """
  #return power.TFmdm_onek_mpc(k)
  return power.TFmdm_onek_mpc(k)
  

def growth_factor_D(z=0.0,cosm=Cosmology()):
  """call to cosmolopy perturbation.py script
  Calculates growth factor """
  
  Theta = 2.728 / 2.7
  #zeq = 2.50 * 10**4 * (cosm.O_m0+cosm.O_de0) * cosm.h_0**2 * Theta**-4
  
  return pert.fgrowth(z,cosm.O_m0)
  

def power_spectrum_P(k,z=0.0,cosm=Cosmology(),camb=False,Tk=0.0):
  """ returns the power spectrum P(k)"""
  
  # WMAP7 parameters:
  #delta_h = 2.43 * (10**-9)
  
  n = 0.966
  
  delta_h = 1.94 * (10**-5) * cosm.O_m0**(-0.785 - (0.05*log(cosm.O_m0))) * exp(-0.95*(n-1)-0.169*(n-1)**2)
  
  if not camb:
    Tk = transfer_function_EH(k)
  
  Dz = growth_factor_D(z)
  
  # speed of light in Mpc s^-1
  c_l = const.c_light_Mpc_s
  
  return (delta_h**2 * 2. * pi**2. * k**n) * (c_l/(cosm.h_0 * const.H100_s))**(3.+n) * (Tk*Dz)**2
  

if __name__ == "__main__":
  k = 0.238961
  r = 10 #Mpc
  
  """
  Tk = transfer_function_EH(k)
  Dz = growth_factor_D(z=3.0)
  Pk = power_spectrum_P(k)
  
  print "T(k) = %s" % Tk
  print "D(z) = %s" % Dz
  print "P(k) = %s" % Pk
  """
  
  Pk = []
  Tk = []
  Dz = []
  
  k_array, T_array = import_transfer_function()
  
  #for k,T in itertools.izip(k_array,T_array):
    #Pk.append(power_spectrum_P(k,camb=True,Tk=T))
  
  
  krange = numpy.logspace(-4,2,1000)
  
  for k in krange:
    Pk.append(power_spectrum_P(k))
    #Tk.append(transfer_function_EH(k))
    #Dz.append(growth_factor_D(z=k))
  
  
  #plt.plot(krange,Pk)
  plt.plot(k_array,T_array,krange,Pk)
  
  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')
  
  plt.show()
  

