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


def import_transfer_function():
  #import a transfer function from a CAMB produced output file
  return NONE

def transfer_function_EH(k,z=0.0,cosm=Cosmology()):
  # set cosmology
  power.TFmdm_set_cosm(cosm.O_m0,cosm.O_b0,-1,0,cosm.O_de0,cosm.h_0,z)
  
  # Call to EH power.c script
  # Calculates transfer function given wavenumber
  return power.TFmdm_onek_mpc(k)
  

def growth_factor_D(z=0.0,cosm=Cosmology()):
  # set cosmology
  power.TFmdm_set_cosm(cosm.O_m0,cosm.O_b0,-1,0,cosm.O_de0,cosm.h_0,z)
  
  # call to cosmolopy perturbation.py script
  # Calculates growth factor 
  return pert.fgrowth(z,cosm.O_m0,unnormed=False)
  

def power_spectrum_P(k,Tk,Dz,z=0.0,cosm=Cosmology()):
  # WMAP7 parameters:
  delta_h = 2.43 * (10**-9)
  n = 0.966
  
  return ((delta_h**2)*2*(pi**2)*(k**n)) * (const.c_light_Mpc_s/cosm.h_0)**(3+n) * (Tk/Dz)**2

if __name__ == "__main__":
  Tk = transfer_function_EH(0.23861)
  Dz = growth_factor_D(z=3.0)
  Pk = power_spectrum_P(0.23861,Tk,Dz)
  
  print "T(k) = %s" % Tk
  print "D(z) = %s" % Dz
  print "P(k) = %s" % Pk
  
  








