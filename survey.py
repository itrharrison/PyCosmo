"""
(HOPEFULLY) USEFUL OBSERVATIONAL SURVEY CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""


import numpy as np
from cosmology import *
from scipy import integrate

class Survey:
  
  def __init__ (self, zmin=0.e0, zmax=2.e0, lnmmin=log(5.e14), lnmmax=log(1.e16), fsky=1.e0):
    """Constructor.
    Default survey sensitive between z=[0,2] and m=[5.e14, 1.e16] and covering
    entire sky
    """
    
    self.z_min = zmin
    self.z_max = zmax
    self.f_sky = fsky
    self.lnm_min = lnmmin
    self.lnm_max = lnmmax
    
  def N_in_survey(self, cosm):
    """Calculate total number of haloes expected to exist within the
    observational survey window.
    
    """
    return self.f_sky*integrate.dblquad(cosm.dNdlnmdz, self.z_min, self.z_max, lambda lnm: self.lnm_min, lambda lnm: self.lnm_max)[0]
