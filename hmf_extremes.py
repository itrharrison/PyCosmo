"""
Code for reproducing EVS of the z=0 halo mass function
(as in Harrison & Coles 2011). New python code to supercede the old (finicky)
C++ one.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

from hmf import *
from cosmology import *
from survey import *

def evs_hypersurface_pdf(r_box=20.e0,
                         lnm_min=log(1.e12),
                         lnm_max=log(1.e18),
                         redshift=0.e0,
                         cosm=Cosmology(),
                         lnm_steps=200):
  """Calculate Extreme Value Statistics for dark matter haloes in a given
  cosmology on a specified spatial hypersurface (constant z box).
  
  Uses the method described in Harrison & Coles 2011 arXiv:0000.0000
  
  Returns
  -------
  phi_max : The EVS pdf
  lnm_arr : The x-points for the pdf
  
  """

  lnm_arr = np.linspace(lnm_min, lnm_max, lnm_steps)
  v_box = (4.e0/3.e0)*pi*r_box**3.e0
  dlnm = abs(lnm_arr[0] - lnm_arr[1])

  f_lon = cosm.dndlnm(lnm_arr, redshift)

  # the underlying cdf
  F = integrate.cumtrapz(f_lon, lnm_arr)

  # total number of haloes
  n_tot = integrate.trapz(f_lon, lnm_arr)
  N = v_box*n_tot

  # normalise
  f = f_lon[:-1] # cumtrapz returns short array
  f = f/n_tot
  F = F/n_tot
  
  # calculate EVS pdf
  Phi_max = F**N
  #phi_max = N*f*pow(F, N - 1.e0) # better to do it this way
  phi_max = np.gradient(Phi_max, dlnm) # ...but this matches DDCSP nicer.
  
  return phi_max, lnm_arr
  
def evs_bin_pdf(z_min=0.e0, z_max=1.e0, z_steps=200,
                lnm_min=log(1.e14), lnm_max=log(1.e17), lnm_steps=200,
                cosm=Cosmology(),
                fsky=1.e0):
  """Calculate extreme value statistics of cold dark matter haloes in a
  given mass and redshift bin.
  
  Uses the method described in Harrison & Coles 2012 arXiv:1111.1184
  
  Returns
  -------
  phi_max : The EVS pdf
  lnm_arr : The x-points for the pdf
  
  """
  
  lnm_arr = np.linspace(lnm_min, lnm_max, lnm_steps)
  dlnm = abs(lnm_arr[0] - lnm_arr[1])
  F = np.zeros_like(lnm_arr)
  
  for i in np.arange(lnm_steps):
    F[i] = cosm.computeNinBin(z_min, z_max, lnm_min, lnm_arr[i])
    
  f = np.gradient(F,dlnm)
                
  N = cosm.computeNinBin(z_min, z_max, lnm_min, lnm_max)
  
  f = f*fsky/N
  F = F*fsky/N
  
  # calculate EVS pdf
  phi_max = N*f*pow(F, N - 1.e0)
  
  return phi_max, lnm_arr

def evs_survey(surv=Survey(), cosm=Cosmology(),
               n_bins=100,
               CLs=(66.e0,95.e0,99.e0)):
  """Produce M_max vs z plot for a given survey, cosmology and
  number of z bins.
  
  Uses the method described in Harrison & Coles 2012 arXiv:1111.1184
  
  Parameters
  ----------
  n_bins : number of redshift bins
  CLs : tuple of requested confidence regions
  
  Returns
  -------
  FIXME!

  """
  
  return 0

if __name__ == '__main__':

  fnl_range = [0,100,300]
  r_range = [20, 50, 100]
  
  st = Hmf(mf_type='st')
  
  wmap7_st = Cosmology()
  wmap7_st.set_hmf(st)
  wmap7_st.display()  
  
  plt.figure(1)
  for fnl in fnl_range:
    phi_max, lnm_arr = evs_hypersurface_pdf(cosm=Cosmology(f_nl=fnl))
    plt.loglog(exp(lnm_arr[:-1]), phi_max, label='$f_{NL} = %d$'%fnl)
  plt.xlabel('Mass $[M_{\odot}h^{-1}]$')
  plt.ylabel('$\phi(M_{max})$')
  plt.ylim([1.e-4,3])
  plt.legend(loc='best')
  
  plt.figure(2)
  for r in r_range:
    phi_max, lnm_arr = evs_hypersurface_pdf(r_box=r)
    plt.loglog(exp(lnm_arr[:-1]), phi_max, label='$r = %d h^{-1}$ Mpc'%r)
  plt.xlabel('Mass $[M_{\odot}h^{-1}]$')
  plt.ylabel('$\phi(M_{max})$')
  plt.ylim([1.e-4,3])
  plt.legend(loc='best')
  
  """
  plt.figure(3)  
  z_arr = linspace(0.e0, 2.e0, 100)
  phi_max_arr = zeros([100,200])
  for i in np.arange(1,len(z_arr)):
    phi_max_arr[i], lnm_arr = evs_bin_pdf(z_arr[i-1], z_arr[i])
    print(i)
    
  plt.semilogx(exp(lnm_arr), phi_max_arr[0])
  plt.xlabel('Mass $[M_{\odot}h^{-1}]$')
  plt.ylabel('$\phi(M_{max})$')  
  plt.show()
  """
