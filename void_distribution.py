"""
Void Distribution Script

Python script for reproducing the
number density distribution of voids
with respect to mass and radius.

Also calculates the Extreme Value 
Statistics of maximum and minimum
radius voids.
"""

import math
import numpy as np
from numpy import sqrt, log, exp, fabs, pi, sin, cos
from matplotlib import pyplot as plt
from scipy import integrate

from powspec import PowSpec
from cosmology import Cosmology


def void_and_cloud(void_barrier, collapse_barrier):
  """The Void-and-Cloud parameter
  
  Parameters
  ----------
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----
  Calculates D, the void-and-cloud parameter, which parametrises 
  the impact of halo evolution on the evolving population of voids
  """
  return fabs(void_barrier) / (collapse_barrier + fabs(void_barrier))
  

def multiplicity_function_svdw(nu,D,void_barrier,collapse_barrier):
  """Multiplicity function defined in Sheth & van de Weyagert (2004 MNRAS 350 517)
  
  Parameters
  ----------
  nu : array
  scaled void size/mass
  
  D : float
  void-and-cloud parameter
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----  
  calculates equation [4] in Sheth & van de Weygaert
  approximating the infinite series in equation [1]
  """
  
  return (1/nu) * (nu/(2*pi))**0.5 * exp(-0.5*nu) \
  * exp((-1*fabs(void_barrier)*(D**2) \
  / (fabs(collapse_barrier)*nu*4)) - (2*(D**4)/(nu**2)))
  

def multiplicity_function_jlh(sigma,D,void_barrier,collapse_barrier):
  """Multiplicity function approximation defined in Jennings, Li & Hu (2013 MNRAS 000 1)
  
  Parameters
  ----------
  sigma : array
  standard deviation of the power spectrum
  
  D : float
  void-and-cloud parameter
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----  
  calculates equation [8] in Jennings, Li & Hu
  approximating the infinite series in equation [6]
  """ 
  x = ((D*sigma)/fabs(void_barrier))
  
  if x <= 0.276:
    return (2./pi)**0.5 * (fabs(void_barrier)/sigma) * exp((-1*void_barrier**2)/(2.*sigma**2))
    
  else:
    summation = 0.0
    for j in range(1,5):
      summation = summation + (exp(-0.5*(j*pi*x)**2) *j*pi*x**2.*sin(j*pi*D))
    
    return 2. * summation
  
multiplicity_function_jlh_vec = np.vectorize(multiplicity_function_jlh)

def multiplicity_function_jlh_exact(sigma,D,void_barrier,collapse_barrier):
  """Exact multiplicity function defined in Jennings, Li & Hu (2013 MNRAS 000 1)
  
  Parameters
  ----------
  sigma : array
  standard deviation of the power spectrum
  
  D : float
  void-and-cloud parameter
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----  
  calculates equation [6] in Jennings, Li & Hu; the exact 
  """ 
  x = ((D*sigma)/fabs(void_barrier))
  
  summation = 0.0
  for j in range(1,500):
    summation = summation + (exp(-0.5*(j*pi*x)**2) *j*pi*x**2.*sin(j*pi*D))
  
  return 2. * summation
  

def scaled_void_distribution(nu,void_barrier=-2.7,collapse_barrier=1.06):
  """ Scaled distribution of void masses/sizes from Sheth & van de Weyagert (2004 MNRAS 350 517)
  
  Parameters
  ----------
  nu : array
  scaled void mass/size
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----
  Calculates nu given in equation [4]; distribution demonstrated in figure(7)  
  """
  
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  return multiplicity_function_jlh_vec(nu,D,void_barrier,collapse_barrier)
  

def void_radii_dist(r,ps,z=0.0,void_barrier=-2.7,collapse_barrier=1.06):
  """ Number density wrt void radii, Jennings, Li & Hu (2013 MNRAS 000 1)
  
  Parameters
  ----------
  r : array
  void radii
  
  ps : object
  PowSpec class instance
  
  z : float
  redshift
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----
  Produces the differential number density of voids 
  wrt to their characteristic radius
  """
  # account for expansion factor (eq.12, Jennings,Li & Hu)
  r = (r / 1.7)
  
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  # calculate volume from a given R
  V = (4. * pi * pow(r,3)) / 3.
  
  # get sigma from PowSpec class fit
  sigma = ps.sig_fit(log(r*1.7))
  
  # get dln(1/sigma) / dln(r) 
  dlns_dlnr = fabs(ps.dlnsigmadlnr(log(r)))
  
  # calculate f(sigma)
  if(np.isscalar(sigma)):
    fSig = multiplicity_function_jlh(sigma,D,void_barrier,collapse_barrier)
  else:
    fSig = multiplicity_function_jlh_vec(sigma,D,void_barrier,collapse_barrier)
  
  no_dens = (fSig/V) * dlns_dlnr
  
  #plt.loglog(radius,fSig,radius,V,radius,dlns_dlnr)
  #plt.legend(["fSig","V","dlns_dlnr"])
  #plt.show()
  #raw_input()
  
  return no_dens
  

def void_mass_dist(m,ps,cosm,z=0.0,void_barrier=-2.7,collapse_barrier=1.06):
  """ Number density wrt void mass, Jennings, Li & Hu (2013 MNRAS 000 1)
  
  Parameters
  ----------
  m : array
  void masses
  
  ps : object
  PowSpec class instance
  
  cosm : object
  Cosmology class instance
  
  z : float
  redshift
  
  void_barrier : float
  the critical underdensity; defines a void
  
  collapse_barrier : float
  the critical overdensity; boundary at which virialized objects are formed
  
  Notes
  -----
  produces the differential number density of voids
  wrt their characteristic mass
  """
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  # Average density of the universe
  rho = cosm.rho_m(z)
  
  # convert mass -> radius,use to get sigma from PowSpec class fit
  r = ( ((3*m)/(4*pi*rho))**(0.3333333333333333333) ) / 1.7
  sigma = ps.sig_fit(log(r*1.7))
  
  # get dln(1/sigma) / dln(m) 
  dlns_dlnm = fabs(ps.dlnsigmadlnm(log(m)))
  
  # calculate f(sigma)
  if(np.isscalar(sigma)):
    fSig = multiplicity_function_jlh(sigma,D,void_barrier,collapse_barrier)
  else:
    fSig = multiplicity_function_jlh_vec(sigma,D,void_barrier,collapse_barrier)
  
  no_dens = ((fSig*rho)/m) * dlns_dlnm
  
  #plt.loglog(m,fSig,m,dlns_dlnm)
  #plt.legend(["fSig","dlns_dlnm"])
  #plt.show()
  #raw_input()
  
  return no_dens


def void_fr(norm,r,ps):
  """ Original void distribution probability density function (pdf)
  
  Parameters
  ----------
  norm : float
  normalisation factor
  
  r : array
  void radii
  
  ps : object
  PowSpec class instance
  
  Notes
  -----
  f(r) from Harrison & Coles (2012); pdf of the original void distribution
  """
  
  num = void_radii_dist(r,ps)
  return (1/norm)*num*(1/r)
  

def void_Fr(norm,r,ps,max_record=True):
  """ Cumulative void distribution
  
  Parameters
  ----------
  norm : float
  normalisation factor
  
  r : array
  void radii
  
  ps : object
  PowSpec class instance
  
  max_record : bool
  direction of cumulative sum
  
  Notes
  -----
  F(r) from Harrison & Coles (2012); known distribution of void radii 
  
  for max_record=True, calculates the cumulative  distribution *upto*
  a given radii, otherwise calculates EVS for small radii
  """
  
  if max_record:
    integ = integrate.quad(void_radii_dist,0.,r,args=(ps))
  else:
    integ = integrate.quad(void_radii_dist,r,np.inf,args=(ps))
  
  return (1/norm)*integ[0]
  

def void_pdf(r,norm,ps,V,max_record=True):
  """ Void probability density function (pdf)
  
  Parameters
  ----------
  r : array
  void radii
  
  norm : float
  normalisation factor
  
  ps : object
  PowSpec class instance
  
  V : float
  Constant redshift box volume
  
  max_record : bool
  direction of cumulative sum
  
  Notes
  -----
  phi(max) from Harrison & Coles (2012); exact extreme value pdf of the
  original void distribution for a given radius
  
  max_record passed to void_Fr function. Further information on its
  ue provided there
  """
  
  fr = void_fr(norm,r,ps)
  Fr = void_Fr(norm,r,ps,max_record)
  N = norm * V
  
  return N * fr * Fr**(N-1)
  

def void_norm(ps):
  """ Void Normalisation factor
  
  Parameters
  ----------
  
  ps : object
  PowSpec class instance
  
  Notes
  -----
  n_tot from Harrison & Coles (2012)
  normalisation factor; gives the 
  total comoving number density of voids """
  
  return integrate.quad(void_radii_dist,0.,np.inf,args=(ps))
  


if __name__ == '__main__':
  # initialise radius and mass lists
  radius = np.logspace(-1,2,600)
  mass = np.logspace(8,17,700)
  radius.tolist()
  mass.tolist()
  
  # initialise cosmology and PowSpec void parameters
  cosm = Cosmology()
  cosm.pk.vd_initialisation(0.0,radius,mass)
  
  # calculate the number density of voids wrt radius
  no_radius = void_radii_dist(radius,cosm.pk)
  #no_radius2 = void_radii_dist(radius,cosm.pk.sigmar,cosm.pk,collapse_barrier=1.686)
  
  # calculate the number density of voids wrt mass
  #no_mass = void_mass_dist(mass,cosm.pk,cosm)
  
  plt.loglog(radius,no_radius)
  plt.show()
  raw_input()
  
  pdf_large = []
  pdf_small = []
  norm = void_norm(cosm.pk)
  
  for r in radius:
    pdf_large.append(void_pdf(r,norm[0],cosm.pk,1000000.))
    pdf_small.append(void_pdf(r,norm[0],cosm.pk,1000000.,max_record=False))
  
  plt.xlim([3e-1,3e1])
  plt.ylim([1e-7,1e-0])
  plt.loglog(radius,pdf_large,radius,pdf_small)
  plt.show()

  """
  ps = PowSpec()
  
  nod = []
  nod2 = []
  
  for R in nu_range:
    nod.append(void_radii_distribution(R,void_barrier=-2.7,ps=ps))
    nod2.append(void_radii_distribution(R,collapse_barrier=1.686,void_barrier=-2.7,ps=ps))
    
  plt.plot(nu_range,nod,nu_range,nod2)
  
  plt.yscale('log')
  plt.xscale('log')
  #plt.yscale('linear')
  #plt.xscale('linear')  
  
  plt.xlabel(r'r [Mpc/h]', fontsize='20')
  plt.ylabel(r'dn/dlnr $(h/Mpc)^{3}$', fontsize='20')
  plt.legend((r'$\delta_{c}=1.06$',r'$\delta_{c}=1.69$'), prop={'size':20})
  
  plt.show()
  """
  """
  nu_range = np.arange(0.1,20,0.1)
  
  fnu1 = []
  fnu2 = []
  fnu3 = []
  
  D = void_and_cloud(void_barrier=-2.81,collapse_barrier=1.06)
  
  for nu in nu_range:
    fnu1.append(scaled_void_distribution(nu,collapse_barrier=1.06))
    fnu2.append(scaled_void_distribution(nu,collapse_barrier=1.69))
    fnu3.append(scaled_void_distribution(nu,collapse_barrier=99999999))
  
  plt.plot(nu_range,fnu1,nu_range,fnu2,nu_range,fnu3)
  plt.legend((r'$\delta_{c}=1.06$',r'$\delta_{c}=1.69$',r'$\delta_{c}=\infty$'), prop={'size':20})
  plt.ylim(0.01,0.6)
  plt.xlim(0.,5)
  #plt.xscale('log')
  #plt.yscale('log')
  plt.yscale('linear')
  plt.xscale('linear')  
  plt.xlabel(r'$\nu = (\delta_{V} / \sigma)^{2}$', fontsize='20')
  plt.ylabel(r'$f(\nu)$', fontsize='20')
  
  plt.show()
"""
