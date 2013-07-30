"""
(MILDLY) USEFUL COSMOLOGY CLASS.
NOTE ALL UNITS ARE WRT h^-1

Much of the cosmography is from (as ever) Hogg arXiv:astro-ph/9905116

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

import numpy as np
from numpy import sqrt, log, exp, fabs, pi
from scipy import integrate

from matplotlib import pyplot as plt

from constants import *
import hmf
from powspec import PowSpec

class Cosmology:
  
  def __init__(self, dc=1.686e0, h0=0.702e0, om=0.274e0, ode=0.725e0,
               w0=-1.e0, ob=0.0458, o_r=8.6e-5, o_k=0.e0, f_nl=0.e0,
               tau_r=0.087, z_r=10.4, ns=0.96,
               hmf_type='tinker'):
    """
    Constructor.
    Default initialisation with WMAP7+BAO+H0 parameters,
    Tinker mass function and fitted P(k).
    """
    
    print "Cosmology"
    
    self.delta_c = dc
    self.h_0=h0
    self.O_m0=om
    self.O_de0=ode
    self.w_0=w0
    self.O_b0=ob
    self.O_r0=o_r
    self.O_k0=o_k
    self.fnl=f_nl
    self.tau_rec=tau_r
    self.z_rec=z_r
    self.eta_r=self.eta(self.z_rec)
    self.H_0 = h0*100.e0
    self.n_s = ns
    
    self.mf = hmf.Hmf(mf_type=hmf_type)
    
    import inspect
    print inspect.getframeinfo(inspect.currentframe().f_back)[3]
    #print inspect.stack()
    
    self.pk = PowSpec(self)
  
  def display(self):
    """
    Displays what you're working with.
    """
    print("YOU ARE USING THE FOLLOWING COSMOLOGY:")
    print("{h_0, O_m, O_de, w_0, O_b, O_r, O_k} = ")
    print("{{{}, {}, {}, {}, {}, {}, {}}}".format(self.h_0, self.O_m0, self.O_de0, self.w_0, self.O_b0, self.O_r0, self.O_k0))
    print("WITH MASS FUNCTION:")
    self.mf.display()
    print("WITH POWER SPECTRUM:")
    self.pk.display()
  
  def set_hmf(self, set_mf):
    """
    Set method for the HMF within the Cosmology
    """
    self.mf = set_mf
    
  def set_powspec(self, set_pk):
    """
    Set method for power spectrum within the Cosmology
    """
    self.pk = set_pk
  
  def rho_c(self):
    """
    Critical density
    """
    return rhofactor * ((3.e0*100.e0*100.e0) / (8.e0*pi*G*Hfactor*Hfactor))
  
  def rho_m(self, z):
    """
    Average density of Universe at redshift z
    """
    return (rhofactor*self.O_m(z) *
            ((3.e0*100.e0*100.e0) / (8.e0*pi*G*Hfactor*Hfactor)))
    
  def O_m(self, z):
    """
    Omega matter.
    """
    return (self.O_m0*pow(1.e0+z, 3.e0) / 
           (self.O_m0*pow(1.e0+z, 3.e0) + 1.e0 - self.O_m0));
  
  def h(self, z):
    """
    Hubble function ***little*** h(z).
    """
    a = 1.e0/(1.e0+z)
    h_square = (self.O_r0/pow(a, 4.e0) +
                self.O_m0/pow(a, 3.e0) +
                self.O_k0/pow(a, 2.e0) +
                self.O_de0/pow(a, 3.e0*(1.e0+self.w_0)))
    
    return sqrt(h_square)
    
  def _integ_one_on_h_scalar(self, z_min, z_max):
    """Scalar calculation of 1/h between z_min and z_max
    """
    integfunc = lambda x : 1.e0/self.h(x)
    return integrate.quad(integfunc, z_min, z_max)[0]
    
  _integ_one_on_h_vector = np.vectorize(_integ_one_on_h_scalar)
  
  def H(self, z):
    """
    Hubble function ***upper*** H(z).
    """
    return self.H_0*self.h(z)
  
  def dvdz(self, z):
    """
    Comoving volume element at redshift z.
    """
    return 4.e0*2998e0*(1+z)*(1+z)*pi*pow(self.D_a(z),2.e0) / self.h(z)
    
  def V_between(self, z_min, z_max):
    """
    Volume between two redshifts.
    """
    points = 200
    int_arr = np.zeros(points)
    z_arr = np.linspace(z_min, z_max, points)
    for i in np.arange(points):
      int_arr[i] = self.dvdz(z_arr[i])
    
    return integrate.trapz(int_arr, z_arr)
  
  def D_a(self, z):
    """
    Angular diameter distance to redshift z.
    """
    if np.isscalar(z):
      integ = self._integ_one_on_h_scalar(0.e0, z)
    else:
      integ = self._integ_one_on_h_vector(self, 0.e0, z)

    return 2998.e0*integ/(1.e0 + z)

  def D_l(self, z):
    """
    Luminosity distance to redshift z.
    """    
    return D_a(z)*(1.e0 + z)**2.e0
    
  def D_c(self, z):
    """
    Comoving radial distance to redshift z.
    """
    return D_a(z)*(1.e0 + z)/(100.e0*self.h0)
    
  def dist_mod(self, z):
    """
    Distance modulus.
    5 * log(D_l(z)) + 25
    """
    return 5.e0*np.log10(self.D_l(z)) + 25.e0
  
  def eta(self, z):
    """
    Size of particle horizon at redshift z
    """  
    if (z == np.inf):
      retVar = 0.e0
    else:
     if np.isscalar(z):
        retVar = self._integ_one_on_h_scalar(z, np.inf)/(100.e0*self.h_0)
     else:
        retVar = self._integ_one_on_h_vector(self, z, np.inf)/(100.e0*self.h_0)
    return retVar
    
  def _integ_lookback_scalar(self, z):
    """Scalar calculation of lookback time to redshift z
    """
    integfunc = lambda x : 1.e0/(self.h(x)*(1.e0 + x))
    return integrate.quad(integfunc, 0.e0, z)[0]/(100.e0*self.h_0)

  _integ_lookback_vector = np.vectorize(_integ_lookback_scalar)
      
  def t_lookback(self, z):
    """Lookback time to redshift z
    """
    if np.isscalar(z):
      t_lb = self._integ_lookback_scalar(z)
    else:
      t_lb = self._integ_lookback_vector(self, z)
    return t_lb
    
  def _integ_growth_scalar(self, z):
    """Scalar calculation of linear growth function
    """
    integfunc = lambda x : 1.e0/pow(self.h((1.e0/x) -1.e0)*x, 3.e0)

    return ((5.e0/2.e0) * self.O_m0 * self.h(z) *
            integrate.quad(integfunc, 0.e0, 1.e0/(1.e0+z))[0])
            
  _integ_growth_vector = np.vectorize(_integ_growth_scalar)
  
  def growth(self, z):
    """Linear growth function.
    """    
    if np.isscalar(z):
      Dplus = self._integ_growth_scalar(z)
    else:
      Dplus = self._integ_growth_vector(self, z)
    return Dplus
    
  
  def dndlnm(self, lnm, z):
    """
    Comoving number density of dark matter haloes in logarithmic m.
    """
    s = self.pk.sigma(lnm)
    dlnsdlnm = self.pk.dlnsigmadlnm(lnm)
    D_z = self.growth(z)
    D_0 = self.growth(0.e0)
    
    return (self.mf.f_sigma(s*D_z/D_0, z) *
            fabs(log(D_z/D_0) + dlnsdlnm) *
            self.rho_m(z) / (exp(lnm)) *
            self.mf.r_ng(s*D_z/D_0, self.delta_c, self.fnl))
    
  def computeNinBin(self, z_min, z_max, lnm_min, lnm_max):
    """
    Total number of dark matter haloes expected within a given mass,
    redshift bin.
    """
    return integrate.dblquad(self.dNdlnmdz,
                             z_min, z_max,
                             lambda lnm: lnm_min, lambda lnm: lnm_max)[0]
  
  def computeLittleNinZBin(self, lnm_min, lnm_max, z):
    """
    Total number of haloes within a given mass bin at fixed redshift.
    """
    return integrate.quad(self.dndlnm, lnm_min, lnm_max, args=(z))[0]
  
  def dNdlnmdz(self, lnm, z):
    """
    Total number of dark matter haloes at a given redshift.
    Product of dndlnm * dVdz
    """
    return self.dndlnm(lnm, z)*self.dvdz(z)
    
  def dNdlnm0dz(self, lnm, z):
    """
    Total number of dark matter haloes,
    of equivalent mass at redshift zero, at a given redshift.
    Product of dndlnm * dVdz
    """
    return self.dndlnm(lnm, 0.e0)*self.dvdz(z)
  

if __name__ == '__main__':
  #wmap7 = Cosmology()
  
  lnm_arr = np.linspace(log(1.e13), log(1.e18), 100)
  
  #plt.loglog(exp(lnm_arr), wmap7.dndlnm(lnm_arr, 0.e0))
  #plt.show()
