from cosmology import *
from math import *
from numpy import *
from scipy import integrate
from numpy import random
from matplotlib import pyplot as pl

f_H = 0.76e0
m_protonkg = 1.673e-27 # in kg
m_proton = 938.272e6 # in keV/c^2
m_electronkg = 9.109e-31 # in kg
m_electron = 510.998 # in keV/c^2
sigma_T = 6.65246e-29/(1.e6*pc*1.e6*pc) # in Mpc^2(!)
#c = 2.998e8

class clust:

  def __init__(self, lnm, redshift, cosm):
    
    self.lnm = lnm
    self.z = redshift
    self.cosm = cosm
    self.clpix = 128
    
    self.deltav = 200.e0
    
    self.zeta = pow(1.e0 + self.z, 1.e0/5.e0)*0.14
    
    self.r_vir = (9.5103e0/(1.e0+self.z)) * ((cosm.O_m(0.e0)*self.deltav/cosm.O_m(self.z))**(-1.e0/3.e0)) * ((exp(lnm)/(1.e15))**(1.e0/3.e0)) # in Mpc
    self.theta_vir = (self.r_vir / cosm.D_a(self.z)) * 180.e0/pi # in deg
    self.kBTe = (1.e0/0.75)*(1.e0+self.z) * ((cosm.O_m(0.e0)*self.deltav/cosm.O_m(self.z))**(1.e0/3.e0)) * ((exp(lnm)/(1.e15))**(2.e0/3.e0)) # in KeV
    
    self.r_c = self.zeta*self.r_vir # in Mpc
    self.theta_c = (self.r_c / cosm.D_a(self.z)) * 180.e0/pi # in deg
    
    self.N_e = (1.e0 + f_H)*(cosm.O_b0/cosm.O_m(0.e0))*exp(lnm)*msol/(2.e0*m_protonkg)
    self.n_e0 = self.N_e / ( 4.e0*pi*(self.r_c**3.e0)*( 1.e0/self.zeta + arctan(self.zeta) - pi/2.e0 ) )

    self.y_0 = 2.e0 * (self.kBTe/(m_electron)) * sigma_T * self.r_c * self.n_e0 * ( pi/2.e0 - arctan(self.zeta) )
    self.Y = (self.kBTe/(m_electron)) * sigma_T * self.N_e / cosm.D_a(self.z)**2.e0
    
    self.RAfrac = random.rand()
    self.DECfrac = random.rand()
    self.theta_max = self.theta_vir
    
  def make_ymap(self, pixsize=None):
    
    if pixsize == None: # for no argument plot with 128^2 pixels
      self.clpix = 128
      pixsize = self.theta_max / self.clpix
    else:
      self.clpix = int(self.theta_max / pixsize)
    
    if self.clpix%2:
      self.clpix = self.clpix+1
    
    self.ymap_urq = zeros([self.clpix/2.e0, self.clpix/2.e0])
    self.ymap = zeros([self.clpix, self.clpix])
    #print(shape(self.ymap))
    clrange = linspace(0.e0, self.theta_max, self.clpix)
    for i in arange(self.clpix/2.e0):
      for j in arange(self.clpix/2.e0):
        self.ymap_urq[i,j] = self.y(sqrt(clrange[i]**2.e0 + clrange[j]**2.e0))
        
    self.ymap[self.clpix/2.e0-1:-1,0:self.clpix/2.e0] = self.ymap_urq[:,::-1]
    self.ymap[0:self.clpix/2.e0, 0:self.clpix/2.e0] = self.ymap_urq[::-1,::-1]
    self.ymap[0:self.clpix/2.e0, (self.clpix/2.e0-1):-1] = self.ymap_urq[::-1,:]
    self.ymap[self.clpix/2.e0-1:-1, self.clpix/2.e0-1:-1] = self.ymap_urq
    
    self.ymap[np.isnan(self.ymap)] = 0
    
      
  def y(self, theta):
  
    return 2.e0 * (self.kBTe/(m_electron)) * sigma_T * self.r_c * self.n_e0 * arctan(sqrt( (1.e0/self.zeta - (theta/self.theta_c)**2.e0)/(1.e0 + (theta/self.theta_c)**2.e0) ))/sqrt(1.e0 + (theta/self.theta_c)**2.e0)
    
  def display(self, pixsize=None):
    
    self.make_ymap(pixsize=pixsize)
    #print(self.ymap)
    pl.figure()
    pl.imshow(self.ymap, interpolation='nearest', origin='lower')
    pl.show()
    
  def info(self):
    
    print("m = {0:.2e} Msol, z = {1:.2f}".format(exp(self.lnm), self.z))
    
  def trim(self, edge, npix):
    if (edge == 'top'):
      self.ymap = np.delete(self.ymap, s_[npix:], 0)
      self.ypix = self.ypix - abs(npix)
    if (edge == 'bottom'):
      self.ymap = np.delete(self.ymap, s_[:npix], 0)
      self.ypix = self.ypix - abs(npix)
    if (edge == 'right'):
      self.ymap = np.delete(self.ymap, s_[npix:], 1)
      self.xpix = self.xpix - abs(npix)
    if (edge == 'left'):
      self.ymap = np.delete(self.ymap, s_[:npix], 1)
      self.xpix = self.xpix - abs(npix)
