"""
(HOPEFULLY) USEFUL POWER SPECTRUM CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

import numpy as np

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
