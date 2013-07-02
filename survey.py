from numpy import *
from cosmology import *
from scipy import integrate

class Survey:
	
	def __init__ (self, zmin=0.e0, zmax=2.e0, lnmmin=log(5.e14), lnmmax=log(1.e16), fsky=1.e0):
		
		self.z_min = zmin
		self.z_max = zmax
		self.f_sky = fsky
		self.lnm_min = lnmmin
		self.lnm_max = lnmmax
		
	def N_in_survey(self, cosm):
		
		return self.f_sky*integrate.dblquad(cosm.dNdlnmdz, self.z_min, self.z_max, lambda lnm: self.lnm_min, lambda lnm: self.lnm_max)[0]
