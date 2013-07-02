"""
Python implementation of Geraint Pratten's code
for generating flat-sky CMB maps...
"""

import numpy as np
from numpy import sqrt, log, exp, fabs, pi
from matplotlib import pyplot as pl
from scipy import misc, random, interpolate, interp
from scipy import fftpack as ft

from cosmology import *

class CMBFlatMap:

  def __init__(self, mapsize=10.e0, pixels=1024, cosm=Cosmology()):
    """
    Constructor.
    Default will create a 10x10 degree FOV with 1024 pixels and
    a WMAP7 Cosmology.
    """
    self.cosm = cosm
    
    self.mapsize_deg = mapsize                      # map size in np.real domain
    self.mapsize_rad = np.deg2rad(self.mapsize_deg)
    self.fsky = (mapsize**2.e0) / 41253.e0
    self.npix = pixels
    
    self.Fmapsize = 1.e0/self.mapsize_rad           # map size in Fourier domain
    
    self.pixsize = self.mapsize_rad / self.npix     # pixel size in radians
    self.pixsize_deg = self.mapsize_deg / self.npix # pixel size in degrees
    
    self.mapaxis = ((self.mapsize_deg / self.npix) * # range of axes
                    np.arange(-self.mapsize_deg/2.e0, self.mapsize_deg/2.e0, 1))
    self.Fmapaxis = 1.e0/self.mapaxis
    
    self.krange = (self.Fmapsize *                   # define k-space
                   np.arange(-self.npix/2.e0, self.npix/2.e0, 1))
    self.kx, self.ky = np.meshgrid(self.krange, self.krange)
    self.modk = sqrt(self.kx**2.e0 + self.ky**2.e0)
        
    self.Txy = random.normal(0,1,[self.npix,self.npix]) # Gaussian random field
    self.Txy = self.Txy - np.mean(self.Txy)
    self.Fxy = ft.fftshift(ft.fft2(self.Txy))           # Fourier domain GRF
    self.Fxy = self.Fxy/sqrt(np.var(self.Fxy))
    
    self.build_Pk(self.cosm)                  # Get flat-sky P(k) for cosmology
    self.Fxy = self.Fxy * self.Pk             # Apply the power spectrum
    self.Txy = np.real(ft.ifft2(ft.fftshift(self.Fxy)))
    
    self.ymap = np.zeros([self.npix, self.npix])
    
  def display(self):
    """
    Plot the flat-sky CMB.
    """
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = pl.imshow(self.Txy, interpolation='nearest', origin='lower',
                    extent=[0,self.mapsize_deg,0,self.mapsize_deg])
    pl.clim(-6.e-5, 6.e-5)
    cb = fig.colorbar(cax, ticks=[-6.e-5,-3.e-5,0,3.e-5,6.e-5])
    cb.ax.set_yticklabels([-6,-3,0,3,6])
    cb.set_label('T ($\mu K$)')
    pl.xlabel('RA (deg)')
    pl.ylabel('DEC (deg)')
    pl.show()
    
  def build_Pk(self, cosm):
    """
    Calculate correct flat-sky P(k) from external Cls (angular power spectrum).
    """
    cmbdata = np.loadtxt(cosm.pk.clfile)
    self.l = cmbdata[:,0]
    self.temp = cmbdata[:,1]
    self.k = self.l / (2.e0*pi)                     # Limber appx.
    
    self.Cl = self.temp * 2.e0 * pi / (self.l*(self.l+1.e0))
    self.Pk = np.interp(self.modk, self.k, sqrt(self.Cl))
    self.Pk[~np.isfinite(self.Pk)] = 0.e0
    
  def apply_beam(self, beamtype, fwhm):
    """
    Apply a beam function to map.
    Currently 'beamtype' is redundant and a Gaussian beam is always used.
    
    ****fwhm is in degrees****
    """
    fwhm = np.deg2rad(fwhm)
    sigmab = fwhm/sqrt(8.e0*log(2.e0))              # fwhm to sigma

    self.Bl = exp(-self.l*(self.l+1.e0)*sigmab*sigmab/2.e0)
    self.beam = np.interp(self.modk, self.k, self.Bl)
    self.beam[~np.isfinite(self.beam)] = 0.e0
    
    self.Fxy = self.Fxy*self.beam                   # apply the beam to the map
    self.Txy = np.real(ft.ifft2(ft.fftshift(self.Fxy)))
  
  def add_ymap(self, ymap, freq=1.e0):
    """
    Add a Compton Y map (e.g. from LSS tSZ) to the map as temperature.
    \DeltaT(x) = -2*T_{CMB}*Y(x)
    """
    self.ymap = ymap
    self.Txy = self.Txy - 2.e0*2.73*ymap
    self.Fxy = ft.fftshift(ft.fft2(self.Txy))
  
  def add_cluster_ymap(self, cl):
    """
    Add Compton Y from an object of the Cluster class to the map as temp.
    TO FIX: edges.
    """
    xcoord = int(cl.RAfrac*self.npix)               # fix edges please!
    ycoord = int(cl.DECfrac*self.npix)
    
    cl.make_ymap(pixsize=self.pixsize_deg)          # calculate the cluster Y
    
    # test for edges
    npix_out_top = (ycoord + cl.clpix/2.e0) - self.npix   # prob if > 0
    npix_out_bottom = (ycoord - cl.clpix/2.e0)            # prob if < 0
    npix_out_right = (xcoord + cl.clpix/2.e0) - self.npix
    npix_out_left = (xcoord + cl.clpix/2.e0)
    
    if (npix_out_top > 0):
      # trim from top
      cl.trim('top', npix_out_top)
      """
      self.ymap = np.delete(self.ymap, s_[npix_out_top:], 0)
      """
    if (npix_out_bottom < 0):
      # trim from bottom
      cl.trim('bottom', npix_out_bottom)
      """
      self.ymap = np.delete(self.ymap, s_[:npix_out_bottom], 0)
      """
    if (npix_out_right > 0):
      # trim from right
      """
      self.ymap = np.delete(self.ymap, s_[npix_out_right:], 1)
      """
    if (npix_out_left < 0):
      # trim from left
      """
      self.ymap = np.delete(self.ymap, s_[:npix_out_left], 1)
      """
      
    xsize, ysize = shape(self.ymap[xcoord-cl.clpix/2.e0:xcoord+cl.clpix/2.e0,
                         ycoord-cl.clpix/2.e0:ycoord+cl.clpix/2.e0])
                         
    self.ymap[xcoord-cl.clpix/2.e0:xcoord+cl.clpix/2.e0,
              ycoord-cl.clpix/2.e0:ycoord+cl.clpix/2.e0] += cl.ymap[:xsize, :ysize]
    
  def add_noise(self, temp):
    """
    Add per-pixel Gaussian random noise.
    """
    self.noise = random.normal(0,temp,[self.npix,self.npix])
    self.Fnoise = ft.fftshift(ft.fft2(self.noise))
    self.Txy = self.Txy + self.noise
    self.Fxy = ft.fftshift(ft.fft2(self.Txy))
    
    self.Clnoise = ((temp*self.mapsize_rad/self.npix)*self.Bl)**-2.e0
    self.Pknoise = np.interp(self.modk, self.k, self.Clnoise)
    
  def matched_filter(self, noisesigma):
    
    self.Wl = 1.e0/((self.Cl+self.Clnoise)*self.Bl)
    self.Wk = np.interp(self.modk, self.k, self.Wl)/noisesigma
    
    self.snFxy = self.Fxy * self.Wk
    self.snxy = np.real(ft.ifft2(ft.fftshift(self.snFxy)))
    
  def matched_filter_Melin(self, theta_c, xfilt, noisesigma, deltaT0=1.e0):
    """ Matched filter with a spherical beta profile as in SPT paper
    """
    
    self.N_astro = 1.e0/self.Cl                   # CMB power spectrum
    self.N_noise = 1.e0/self.Clnoise              # noise power spectrum
    
    #self.Wl = self.Bl/(self.Bl*self.Bl*self.N_astro + self.N_noise)
    self.Wl = 1.e0/((self.Cl+self.Clnoise)*self.Bl)
    self.Wk = np.interp(self.modk, self.k, self.Wl)/noisesigma
    
    #print(theta_c)
    
    self.Sk = self.beta_profile_map(theta_c)
    
    self.snFxy = self.Fxy * self.Wk * self.Sk
    self.snxy = np.real(ft.ifft2(ft.fftshift(self.snFxy)))
    
  def beta_profile_map(self, theta_crit, cutoff_factor=10.e0, deltaT0=1.e0):
    theta_c = np.deg2rad(theta_crit)
    theta_cutoff = theta_c * cutoff_factor
    
    theta = linspace(-theta_cutoff, theta_cutoff, 128)
    y = deltaT0 / (1.e0 + (abs(theta)/theta_c)**2.e0)
    
    Fy = ft.fftshift(ft.fft(y))
    
    return Fy
    
    """
    #theta_c = np.deg2rad(theta_crit)
    theta_c = np.deg2rad(1.e0/60.e0)
    axis = np.linspace(-self.mapsize_rad/2.e0, self.mapsize_rad/2.e0, self.npix)
    
    theta = lambda x : 1.e0 / (1.e0 + (x/theta_c)**2.e0)
    
    filtermap = np.zeros([self.npix, self.npix])
    for i in np.arange(self.npix):
      for j in np.arange(self.npix):
        filtermap[i,j] = theta(sqrt((axis[i] - posn[0])**2 +
                                    (axis[j] - posn[1])**2))
                                    
    #pl.imshow(filtermap, origin='lower')
    #pl.show()
    
    return filtermap
    """
  """
  def find_peaks(self, neighbourhood, threshold):
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
    
    data_max = filters.maximum_filter(abs(self.snxy), neighbourhood)
    maxima = (abs(self.snxy) == data_max)
    diff = (data_max > threshold)
    maxima[diff == 0] = 0
    
    labeled, numobjects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
      x_centre = (dx.start + dx.stop - 1)/2
      x.append(x_centre)
      y_centre = (dy.start - dy.stop - 1)/2
      y.append(y_centre)
    
    figure()
    imshow(self.ymap)
    clim(-6.e-5, 6.e-5)
    plot(x,y, 'ro', markerfacecolor='none')
    """
