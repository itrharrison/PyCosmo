"""
'powspec_init.py'

Generic Power spectrum calculator

produces a usable power spectrum from CAMB files or the Eisenstein & Hu analytic function

"""

#from ..CosmoloPy.cosmolopy.EH import power as power
#from ..CosmoloPy.cosmolopy.EH import tf_fit as tf_fit

from cosmolopy.EH import power as power
from cosmolopy import perturbation as pert
from cosmology import *

def import_powspec():
  return NONE
  
  
  

def filter_function_W():
  return NONE
  
  

def standard_deviation_sigma():
  return NONE
  
  

def variance_S():
  return NONE
  
  

def transfer_function_EH(k,z=0.0,cosm=Cosmology()):
  # set cosmology
  power.TFmdm_set_cosm(cosm.O_m0,cosm.O_b0,-1,0,cosm.O_de0,cosm.h_0,z)
  
  # calculate transfer function given wavenumber
  return power.TFmdm_onek_mpc(k)
  

def growth_factor_D(k,z=0.0,cosm=Cosmology()):
  # set cosmology
  power.TFmdm_set_cosm(cosm.O_m0,cosm.O_b0,-1,0,cosm.O_de0,cosm.h_0,z)
  
  return pert.fgrowth(z,cosm.O_m0,unnormed=False)
  

if __name__ == "__main__":
  print transfer_function_EH(0.23861)
  
  








