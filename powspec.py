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

import cPickle as pickle

import utils as ut
import EH.power as power
import constants as ct

from matplotlib import pyplot as plt

class PowSpec:

  def __init__(self,cosmology):
    """Constructor


    Default to a polynomial fit to WMAP7+BAO+H0 ML
    parameter power spectrum from CAMB. """

    print "PowSpec"

    #initialise cosmology
    self.cosm = cosmology

    #choose power spectrum source
    camb_choice = self.choose()

    self.sigma = self.sigma_wmap7fit
    self.dlnsigmadlnm = self.dlnsigmadlnm_wmap7fit
    self.label = "fitted to WMAP7->CAMB"
    self.clfile="wmap7_bao_h0_scalCls.dat"


  def choose(self):
    """ Choose between an Eisenstein & HU fitting function or a CAMB power spectrum """
    print "  \n \
            ------------------------------------------------------------- \n \
            Would you like to import a CAMB matter/power spectrum file? \n \
            If not, an analytic spectrum using the prescription of \n \
            Eisenstein & Hu (1999,511) will be used. \n \n \
            If using a CAMB file ensure the cosmological \n \
            parameters used to generate the spectrum are \n \
            identical to those specified within the \n \
            cosmology class. \n \n \
            ------------------------------------------------------------- \n \
            If CAMB, enter 'True', else EH \n \
            ------------------------------------------------------------- \n"

    s = raw_input(":")

    if s == "True":
      return True
    return False


  def display(self):
    """Display method to show power spectrum currently working with.
    """
    print("Power spectrum {}".format(self.label))


  def vd_initialisation(self,rrange,mrange,z=0.0):
    """ initialise parameters required for
        void_distribution.py script """

    self.growth_func(z)
    self.sigma_r(rrange,z)
    self.sigma_fit(rrange,self.sigmar)
    self.dlnsigma_dlnr(rrange,z)
    self.dlnsigma_dlnm(mrange,z)

    return None


  def growth_func(self,z):
    """ initialises growth function variable
        as part of PowSpec instance """

    self.Dz = self.cosm.growth(z)
    return self.Dz

  def import_powerspectrum(self,ident,z=0.0):
    """ import power spectrum function from
        a CAMB produced output file """

    z=str(int(z))

    k_array = []
    P_array = []

    for line in open('camb/camb_{0}_matterpower_z{1}.dat'.format(ident,z),'r'):
      a, b = line.split()
      k_array.append(float(a))
      P_array.append(float(b))

    return k_array, P_array

  def interpolate(self,array_1,array_2):
    """ returns a function that uses interpolation
        to find the value of new points """

    return interpolate.interp1d(array_1,array_2)


  def transfer_function_EH(self,k,z):
    """Calculates transfer function given wavenumber"""

    # set cosmology
    power.TFmdm_set_cosm(self.cosm.O_m0,self.cosm.O_b0,\
                        0.0,0,self.cosm.O_de0,self.cosm.h_0,z)

    """Call to EH power.c script
       ???? h Mpc^-1 OR Mpc^-1 ???? """
    #return power.TFmdm_onek_mpc(k)

    return power.TFmdm_onek_hmpc(k)


  def power_spectrum_P(self,k,z):
    """ returns the power spectrum P(k)"""

    delta_h = 1.94 * (10**-5) * self.cosm.O_m0**(-0.785 - (0.05*log(self.cosm.O_m0))) \
               * exp(-0.95*(self.cosm.n_s-1)-0.169*(self.cosm.n_s-1)**2)

    Tk = self.transfer_function_EH(k,z)

    c_l = ct.const["c"] / ct.convert["Mpc_m"]   # speed of light in Mpc s^-1

    return (delta_h**2 * 2. * pi**2. * k**self.cosm.n_s) * \
           (c_l/(self.cosm.h_0 * ct.convert['H0']))**(3.+self.cosm.n_s) * (Tk*self.Dz)**2


  def sigma_fit(self,rrange,sigma_r):
    fit = np.polyfit(log(rrange),sigma_r,6)
    self.sig_fit = np.poly1d(fit)

    return self.sig_fit

  def sigma_r(self,r,z):
    """ returns root of the matter variance, smoothed
        with a top hat window function at a radius r """

    if np.isscalar(r):
      s_r_sq, s_r_sq_error = self.sigma_r_sq(r,z)
      self.sigmar = sqrt(s_r_sq) * self.Dz
    else:
      s_r_sq, s_r_sq_error = self.sigma_r_sq_vec(self,r,z)
      self.sigmar = sqrt(s_r_sq) * self.Dz

    self.sigmar.tolist()
    return self.sigmar

  def sigma_r_sq(self,r,z):
    """ integrate the function in sigma_integral
        between the limits of k : 0 to inf. """

    s_r_sq, s_r_sq_error = integrate.quad(self.sigma_integral,0.,np.inf,args=(r,z))#,limit=10000)

    return s_r_sq, s_r_sq_error

  sigma_r_sq_vec = np.vectorize(sigma_r_sq)    #vectorize sigma-squared function

  def sigma_integral(self,k,r,z):
    """ returns the integral required to calculate sigma
        squared (Coles & Lucchin pg.266, A.Zentner 06 eq.14)"""

    return (k**2 / (2 * pi**2)) * fabs(self.tophat_w(k,r)) * self.power_spectrum_P(k,z)

  def tophat_w(self, k, r):
    """ Fourier transform of the real space tophat
        window function (eq.9 from A.Zentner 06) """

    return (3.*(sin(k*r) - k*r*cos(k*r)))/((k*r)**3.)


  def dlnsigma_dlnr(self,rrange,z):
    """ slope of root matter variance wrt log radius:
        d(log(sigma)) / d(log(r))

        Polynomial fit to supplied cosmology.
        Returns poly1d object """

    self.sigma_r(rrange,z)

    fit = np.polyfit(log(rrange),log(self.sigmar),6)
    self.dlnsigmadlnr = np.polyder(np.poly1d(fit))

    return self.dlnsigmadlnr


  def dlnsigma_dlnm(self,mrange,z):
    """ slope of root matter variance wrt log mass:
        d(log(sigma)) / d(log(M))

        Polynomial fit to supplied cosmology.
        Returns poly1d object """

    rho = self.cosm.rho_m(z)

    # convert mass -> radius to get sigma
    rrange = ((3*mrange)/(4*pi*rho))**(0.333333333333333333333)
    self.sigma_r(rrange,z)

    fit = np.polyfit(log(mrange),log(self.sigmar),6)
    self.dlnsigmadlnm = np.polyder(np.poly1d(fit))

    return self.dlnsigmadlnm


  def sigma_wmap7fit(self, lnm):
    """
    Root of matter variance smoothed with top hat window function on a scale
    specified by log(m)

    Polynomial fit to calculation from a CAMB power spectrum
    with WMAP7 parameters
    """
    return np.exp(18.0058 - 1.47523*lnm + 0.0385449*lnm*lnm - 0.0000112539*pow(lnm,4) + (1.3719e-9)*pow(lnm,6))


  def dlnsigmadlnm_wmap7fit(self, lnm):
    """
    Slope of root matter variance wrt log mass:
    d(log(sigma)) / d(log(m))

    Polynomial fit to calculation from a CAMB power spectrum
    with WMAP7 parameters
    """
    return -1.47523 + 0.0770898*lnm - 0.0000450156*pow(lnm,3) + (8.23139e-9)*pow(lnm,5)

if __name__ == "__main__":
  from cosmology import Cosmology

  cosm = Cosmology()
  ps = PowSpec(cosm)

  radius=np.logspace(-1,2,500)
  radius.tolist()

  sigma = ps.sigma_r(radius,0.)
  ps.sigma_fit(radius,sigma)

  plt.plot(radius,sigma,radius,ps.sig_fit(radius))

  """
  from cosmology import Cosmology

  cosm = Cosmology()
  ps = PowSpec(cosm)


  Pk = []
  Pnew = []
  knew = []

  ps.growth_func(0.0)

  k_array, P_array = ps.import_powerspectrum(ident="36837468")

  # find limits of camb power spectrum
  maxk = max(k_array)
  mink = min(k_array)

  # return interpolated function
  fint = ps.interpolate(k_array,P_array)

  krange = np.logspace(-4,2,1000)

  for k in krange:
    Pk.append(ps.power_spectrum_P(k,0.0))
    if k > mink and k < maxk:
      Pnew.append(fint(k))
      knew.append(k)

  #plt.plot(krange,Pk)
  plt.plot(k_array,P_array,knew,Pnew,krange,Pk)

  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')

  plt.show()

  """
  """
  sigmar = []

  rrange = np.arange(0.1,20,0.1)

  for r in rrange:
    sigmar.append(ps.sigma_r(r))

  #plt.plot(rrange,sigmar)

  #plt.show()
  """
  """
  #plt.yscale('linear')
  #plt.xscale('linear')
  plt.yscale('log')
  plt.xscale('log')

  plt.show()
"""



