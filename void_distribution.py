"""
Python script for reproducing the
distribution of number density of voids
"""

import math
import numpy as np
from numpy import sqrt, log, exp, fabs, pi, sin, cos
from matplotlib import pyplot as plt
from scipy import integrate

from powspec import PowSpec
from cosmology import Cosmology


def void_and_cloud(void_barrier, collapse_barrier):
  # Calculates D ; the void-and-cloud parameter
  return fabs(void_barrier) / (collapse_barrier + fabs(void_barrier))
  

def multiplicity_function_svdw(nu,D,void_barrier,collapse_barrier):
  """
  calculates equation (4) in Sheth & van de Weygaert
  approximating the infinite series in equation (1)
  """
  
  return (1/nu) * (nu/(2*pi))**0.5 * exp(-0.5*nu) \
  * exp((-1*fabs(void_barrier)*(D**2) \
  / (fabs(collapse_barrier)*nu*4)) - (2*(D**4)/(nu**2)))
  

def multiplicity_function_jlh(sigma,D,void_barrier,collapse_barrier):
  """
  Jennings, Li & Hu f(lnsigma) approximation
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
  """
  Jennings, Li & Hu f(lnsigma) approximation
  """ 
  x = ((D*sigma)/fabs(void_barrier))
  
  summation = 0.0
  for j in range(1,500):
    summation = summation + (exp(-0.5*(j*pi*x)**2) *j*pi*x**2.*sin(j*pi*D))
  
  return 2. * summation
  

def scaled_void_distribution(nu,void_barrier=-2.7,collapse_barrier=1.06):
  """
  'A Hierarchy of Voids : Sheth & van de Weygaert'
  Reproduces a scaled distribution of void masses/sizes
  shown in figure(7)  
  """
  
  # D ; the void-and-cloud parameter
  D = void_and_cloud(void_barrier, collapse_barrier)
  
  return multiplicity_function_jlh_vec(nu,D,void_barrier,collapse_barrier)
  

def void_radii_dist(r,ps,z=0.0,void_barrier=-2.7,collapse_barrier=1.06):
  """
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
  dlns_dlnr = fabs(ps.p_der(log(r)))
  
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
  


def void_fr(norm,r,ps):
  """
  f(m) from Harrison & Coles (2012)
  - pdf of the original void distribution
  """
  num = void_radii_dist(r,ps)
  return (1/norm)*num*(1/r)
  

def void_Fr(norm,r,ps):
  """
  F(m) from Harrison & Coles (2012)
  - known distribution of void radii
  """
  integ = integrate.quad(void_radii_dist,0.,r,args=(ps))
  return (1/norm)*integ[0]
  

def void_pdf(r,norm,ps,V):
  """
  phi(max) from Harrison & Coles (2012)
  - exact extreme value pdf of 
  the original void distribution
  """
  fr = void_fr(norm,r,ps)
  Fr = void_Fr(norm,r,ps)
  N = norm * V
  
  print fr, Fr, N
  
  return N * fr * Fr**(N-1)
  

def void_norm(ps):
   """
  n_tot from Harrison & Coles (2012)
  
  - normalisation factor; gives the 
  total comoving number density of voids
  """
  return integrate.quad(void_radii_dist,0.,np.inf,args=(ps))
  


if __name__ == '__main__':
  
  radius = np.logspace(-1,1.7,400)
  radius.tolist()
  
  cosm = Cosmology()
  #cosm.pk.growth_func(0.0)
  cosm.pk.vd_initialisation(0.0,radius)
  
  num = void_radii_dist(radius,cosm.pk)
  #num2 = void_radii_dist(radius,cosm.pk.sigmar,cosm.pk,collapse_barrier=1.686)
  
  plt.loglog(radius,num)
  plt.show()
  raw_input()
  
  pdf = []
  norm = void_norm(cosm.pk)
  
  for r in radius:
    print r
    pdf.append(void_pdf(r,norm[0],cosm.pk,1000.))
  
  plt.xlim([3e-1,3e1])
  plt.ylim([1e-7,1e-0])
  plt.loglog(radius,pdf)
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
