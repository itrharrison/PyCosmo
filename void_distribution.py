"""Void Distribution Script

Python script for reproducing the
number density distribution of voids
with respect to mass and radius.

Also calculates the Extreme Value
Statistics of maximum and minimum
radius voids.
"""

import cPickle as pickle

import math
import numpy as np
from numpy import sqrt, log, exp, fabs, pi, sin, cos
from matplotlib import pyplot as plt
from scipy import integrate

from powspec import PowSpec
from cosmology import Cosmology

class Voids:

  def __init__(self,cosmology):

    self.void_barrier = -2.7
    self.collapse_barrier = 1.686


  def void_and_cloud(self):
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
    return fabs(self.void_barrier) / \
           (self.collapse_barrier + fabs(self.void_barrier))


  def multiplicity_function_svdw(self,nu,D):
    """Multiplicity function defined in Sheth &
       van de Weyagert (2004 MNRAS 350 517)

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
    * exp((-1*fabs(self.void_barrier)*(D**2) \
    / (fabs(self.collapse_barrier)*nu*4)) - (2*(D**4)/(nu**2)))


  def multiplicity_function_jlh(self,sigma,D):
    """Multiplicity function approximation defined
       in Jennings, Li & Hu (2013 MNRAS 000 1)

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
    x = ((D*sigma)/fabs(self.void_barrier))

    if x <= 0.276:
      return (2./pi)**0.5 * (fabs(self.void_barrier)/sigma) \
             * exp((-1*self.void_barrier**2)/(2.*sigma**2))

    else:
      summation = 0.0
      for j in range(1,5):
        summation = summation + (exp(-0.5*(j*pi*x)**2) *j*pi*x**2.*sin(j*pi*D))

      return 2. * summation

  multiplicity_function_jlh_vec = np.vectorize(multiplicity_function_jlh)

  def multiplicity_function_jlh_exact(self,sigma,D):
    """Exact multiplicity function defined
       in Jennings, Li & Hu (2013 MNRAS 000 1)

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
    x = ((D*sigma)/fabs(self.void_barrier))

    summation = 0.0
    for j in range(1,500):
      summation = summation + (exp(-0.5*(j*pi*x)**2) *j*pi*x**2.*sin(j*pi*D))

    return 2. * summation


  def multiplicity_function_pls(self,sigma):
    """ multiplicity function for a single crossing barrier from
    Paranjape, Lam & Sheth 2012 """

    return 0.5 * abs(self.void_barrier) * (1/(2*pi)**0.5) \
           * (1/(sigma**3)) * exp((-1 * self.void_barrier**2)/(2*sigma**2))

  multiplicity_function_pls_vec = np.vectorize(multiplicity_function_pls)

  def scaled_void_distribution(self,nu):
    """Scaled distribution of void masses/sizes
       from Sheth & van de Weyagert (2004 MNRAS 350 517)

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
    D = self.void_and_cloud(self.void_barrier,self.collapse_barrier)

    return self.multiplicity_function_jlh_vec(nu,D)


  def void_radii_dist(self,r,ps):
    """ Number density wrt void radii, Jennings, Li & Hu (2013 MNRAS 000 1)

    Parameters
    ----------

    r : array
    void radii

    ps : object
    PowSpec class instance

    z : float
    redshift

    Notes
    -----

    Produces the differential number density of voids
    wrt to their characteristic radius.
    equation [12], Jennings, Li & Hu
    """
    # account for expansion factor (eq.12, Jennings,Li & Hu)
    r = (r / 1.7)

    # D ; the void-and-cloud parameter
    D = self.void_and_cloud()

    # calculate volume from a given R
    V = (4. * pi * pow(r,3)) / 3.

    # get sigma from PowSpec class fit
    sigma = ps.sig_fit(log(r*1.7))

    # get dln(1/sigma) / dln(r)
    dlns_dlnr = fabs(ps.dlnsigmadlnr(log(r)))

    # calculate f(sigma)
    if(np.isscalar(sigma)):
      fSig = self.multiplicity_function_pls(sigma)
    else:
      fSig = self.multiplicity_function_pls_vec(self,sigma)

    no_dens = (fSig/V) * dlns_dlnr

    return no_dens


  def void_radii_dist_linear(self,r,ps):
    """ Number density wrt void radii in the
        linear domain, Jennings, Li & Hu (2013 MNRAS 000 1)

    Parameters
    ----------

    r : array
    void radii

    ps : object
    PowSpec class instance

    z : float
    redshift

    Notes
    -----

    Produces the differential number density of voids
    wrt to their characteristic radius in the liear domain
    (equation [10] Jennings,Li & Hu)
    """
    # D ; the void-and-cloud parameter
    D = self.void_and_cloud()

    # calculate volume from a given R
    V = (4. * pi * pow(r,3)) / 3.

    # get sigma from PowSpec class fit
    sigma = ps.sig_fit(log(r))

    # get dln(1/sigma) / dln(r)
    dlns_dlnr = fabs(ps.dlnsigmadlnr(log(r)))

    # calculate f(sigma)
    if(np.isscalar(sigma)):
      fSig = self.multiplicity_function_jlh(sigma,D)
    else:
      fSig = self.multiplicity_function_jlh_vec(self,sigma,D)

    no_dens = (fSig/V) * dlns_dlnr

    return no_dens


  def void_radii_dist_vdn(self,r,ps):
    """ Volume conserving model for void radii
    relative abundances, Jennings, Li & Hu 2013 """

    # D ; the void-and-cloud parameter
    D = self.void_and_cloud()

    # calculate volume from a given R
    V = (4. * pi * pow(r,3)) / 3.

    # get sigma from PowSpec class fit
    sigma = ps.sig_fit(log(r))

    # get dln(1/sigma) / dln(r)
    dlns_dlnr = fabs(ps.dlnsigmadlnr(log(r/1.7)))

    # calculate f(sigma)
    if(np.isscalar(sigma)):
      fSig = self.multiplicity_function_jlh(sigma,D)
    else:
      fSig = self.multiplicity_function_jlh_vec(self,sigma,D)

    no_dens = (fSig/V) * dlns_dlnr

    return no_dens


  def void_mass_dist(self,m,ps,cosm,z=0.0):
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

    Notes
    -----

    produces the differential number density of voids
    wrt their characteristic mass
    """
    # D ; the void-and-cloud parameter
    D = self.void_and_cloud()

    # Average density of the universe
    rho = cosm.rho_m(z)

    # convert mass -> radius,use to get sigma from PowSpec class fit
    r = ( ((3*m)/(4*pi*rho))**(0.3333333333333333333) ) / 1.7
    sigma = ps.sig_fit(log(r*1.7))

    # get dln(1/sigma) / dln(m)
    dlns_dlnm = fabs(ps.dlnsigmadlnm(log(m)))

    # calculate f(sigma)
    if(np.isscalar(sigma)):
      fSig = self.multiplicity_function_jlh(sigma,D)
    else:
      fSig = self.multiplicity_function_jlh_vec(sigma,D)

    no_dens = ((fSig*rho)/m) * dlns_dlnm

    #plt.loglog(m,fSig,m,dlns_dlnm)
    #plt.legend(["fSig","dlns_dlnm"])
    #plt.show()
    #raw_input()

    return no_dens

  def void_fr(self,norm,r,ps):
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

    num = self.void_radii_dist(r,ps)
    return (1/norm)*num*(1/r)


  def void_Fr(self,norm,r,ps,max_record):
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
      integ = integrate.quad(self.void_radii_dist,0.,r,args=(ps))
    else:
      integ = integrate.quad(self.void_radii_dist,r,np.inf,args=(ps))

    return (1/norm)*integ[0]


  def void_pdf(self,r,norm,ps,V,max_record=True):
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

    fr = self.void_fr(norm,r,ps)
    Fr = self.void_Fr(norm,r,ps,max_record)
    N = norm * V

    return N * fr * Fr**(N-1)


  def void_norm(self,ps):
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

    return integrate.quad(self.void_radii_dist,0.,np.inf,args=(ps))


  def cumulative_V_R(self,r,ps,func):
    """ cumulative fration of volume contained within voids
    from Jennigs, Li & Hu, 2013 """

    def cumulative_int(r,ps,func):
      # integral function argument
      n = func(r,ps)
      V = (4 * pi * r**3) / 3
      return n * V * (1/r)

    return integrate.quad(cumulative_int,r,np.inf,args=(ps,func))

  cumulative_V_R_vec = np.vectorize(cumulative_V_R)

  def conditional_mf_fp(self,halo_m,void_m,cosm,ps,z=0.0,sigmav=False):
    """ Conditional Mass function, Furlanetto & Piran 2008
        equation [5]
        based on the Press & Schechter mass function
    """
    rho = cosm.rho_m(z)

    r = ((halo_m*3)/(4*pi*rho*200))**(0.333333333333333)
    sigma = ps.sig_fit(log(r))

    # void radius for a region with linearized underdensity delta_v
    r = ((void_m*3)/(4*pi*rho*0.2))**(0.333333333333333)
    #if sigmav is False:
    sigmav = ps.sig_fit(log(r))

    n =  (2/pi)**0.5 * (rho/halo_m**2) * abs(ps.dlnsigmadlnm(log(halo_m))) * (sigma**2) \
    * ((self.collapse_barrier-self.void_barrier)/((sigma**2) - (sigmav**2))**1.5) \
    * exp((-1 * (self.collapse_barrier-self.void_barrier)**2)/(2 * ((sigma**2) - (sigmav**2))))

    return n

  def hod_kravtsov(self,halo_m,m_min):
    """ Halo occupation distribution (kravtsov et al. 2004)

    Parameters
    ----------

    halo_m : float
    Halo mass

    C : int
    Halo ocupation model fit parameter

    beta : int
    Halo occupation model fit parameter

    m_min : float
    minimum galaxy mass, as defined by the obervational detection threshold,
    NOT the actual minimum mass of a galaxy (Kravtsov et al. 2004)

    Notes
    -----

    As demonstrated in Furlanetto & Piran, equation [6]. Total occupation
    number composed of the central occupation plus the satellite occupation
    """
    # recommended values for z = 0 from Kravtsov et al.
    C = 30
    beta = 1

    #satellite galaxies
    Ns = (halo_m / (C * m_min))**beta

    # central galaxy occupation number
    Nc = 1.

    return Ns + Nc

  def galaxy_no_density(self,void_m,m_min,cosm,ps,z=0.0):
    """ Total comoving number density of galaxies within a given region
        Furlanetto & Piran 2008
    """

    def integ(halo_m,void_m,m_min,cosm,ps,z):
      #print self.hod_kravtsov(halo_m,m_min)
      #print self.conditional_mf_fp(halo_m,void_m,cosm,ps,z)
      ng = self.hod_kravtsov(halo_m,m_min) * self.conditional_mf_fp(halo_m,void_m,cosm,ps,z)
      #print ng
      return ng

    n_gal, error = integrate.quad(integ,m_min,np.inf,args=(void_m,m_min,cosm,ps,z))

    #print n_gal

    return n_gal

  def galaxy_underdensity(n_gal,n_halo):
    """ total observed galaxy underdensity in a void with physical size Rv
    """
    delta_g = (n_gal/(n_halo * (1.7**3)))-1
    return delta_g

def dump_pickle(cosm):
  pickle.dump(cosm,open("wmap7.p","wb"))

def load_pickle():
  f = open("void_init.p","rb")
  Dz, sigmar, sig_fit, dlnsigmadlnr, dlnsigmadlnm= pickle.load(f)
  f.close()
  return Dz, sigmar, sig_fit, dlnsigmadlnr, dlnsigmadlnm


if __name__ == '__main__':
  cosm = Cosmology()
  vd = Voids(cosm)

  radius = np.logspace(-1,2,600)
  mass = np.logspace(8.1,17,700)
  radius.tolist()
  mass.tolist()

  fig = plt.figure()

  #cosm.pk.vd_initialisation(0.0,radius,mass)

  # load pickled void parameters from powspec class
  cosm.pk.Dz, cosm.pk.sigmar, cosm.pk.sig_fit, cosm.pk.dlnsigmadlnr, cosm.pk.dlnsigmadlnm = load_pickle()

  #no_radius = vd.void_radii_dist(radius,cosm.pk)
  #plt.loglog(radius,no_radius)

  no_mass = []
  no_mass2 = []

  #for m in mass:
    #no_mass.append(
    #print (vd.conditional_mf_fp(m,10**15,cosm,cosm.pk))
    #vd.void_barrier = 0.00
    #no_mass2.append(vd.conditional_mf_fp(m,0.,cosm,cosm.pk,sigmav=0.))
    #no_mass.append(
    #print (vd.hod_kravtsov(m,10**10))
    #no_mass.append(vd.galaxy_no_density(m,10**11,cosm,cosm.pk))

  halo_mass = np.logspace(7,17.5,700)
  halo_mass.tolist()

  gal_nd = []
  gal_nd2 = []
  gal_nd3 = []
  gal_nd4 = []

  ax1 = fig.add_subplot(111)

  for m in halo_mass:
    gal_nd.append(vd.hod_kravtsov(m,10**10) * vd.conditional_mf_fp(m,10**16,cosm,cosm.pk))
    gal_nd2.append(vd.hod_kravtsov(m,10**10) * vd.conditional_mf_fp(m,10**15,cosm,cosm.pk))
    gal_nd3.append(vd.hod_kravtsov(m,10**10) * vd.conditional_mf_fp(m,10**14,cosm,cosm.pk))
    gal_nd4.append(vd.hod_kravtsov(m,10**10) * vd.conditional_mf_fp(m,10**13,cosm,cosm.pk))

  #no_mass = vd.galaxy_no_density(10**16,10**10,cosm,cosm.pk)

  #print no_mass
  ax1.loglog(halo_mass,gal_nd)
  ax1.loglog(halo_mass,gal_nd2)
  ax1.loglog(halo_mass,gal_nd3)
  ax1.loglog(halo_mass,gal_nd4)

  ax1.legend([r"10^{16}",r"10^{15}",r"10^{14}",r"10^{13}"])

  plt.title("Comoving No Density of Galaxies against resident Halo Mass, \n in a Void of given Mass")
  plt.xlabel(r"Halo mass M_{0}")
  plt.ylabel("Galaxy No. Density")
  #plt.yscale((10**-12),1.)
  plt.show()

  """
  #plot
  cumul, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist_vdn)
  vd.collapse_barrier = 1.686
  cumul2, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist_vdn)
  plt.fill_between(radius,cumul,cumul2,alpha=0.5)

  vd.collapse_barrier = 1.06
  cumulb, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist_linear)
  vd.collapse_barrier = 1.686
  cumulb2, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist_linear)
  plt.fill_between(radius,cumulb,cumulb2,alpha=0.5)

  vd.collapse_barrier = 1.06
  cumulc, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist)
  vd.collapse_barrier = 1.686
  cumulc2, error = vd.cumulative_V_R_vec(vd,radius,cosm.pk,vd.void_radii_dist)
  plt.fill_between(radius,cumulc,cumulc2,alpha=0.5)

  plt.xscale('log')
  plt.show()

  """
  """
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
