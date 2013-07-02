"""
(HOPEFULLY) USEFUL HALO MASS FUNCTION CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

import numpy as np
from numpy import sqrt, log, exp, fabs, pi

class Hmf:
	
	def __init__(self, mf_type='tinker', rng_type='pgh'):
		
		if (mf_type=='tinker'):
			self.label = "Tinker (evolving)"
			self.params=[0.186e0, 1.47e0, 2.57e0, 1.19e0]
			self.f_sigma = self.tinker
		elif (mf_type=='st'):
			self.label = "Sheth-Tormen"
			self.params=[0.322, 0.707, 0.3, 1.686]
			self.f_sigma = self.st
		elif (mf_type=='ps'):
			self.label = "Press-Schechter"
			self.params=[1.686]
			self.f_sigma = self.ps
		else:
			print("ERROR uknown hmf \'{}\' selected, please choose from:".format(mf_type))
			print("- Tinker (\'tinker\')")
			print("- Sheth-Tormen (\'st\')")
			print("- Press-Schechter (\'ps\')")
		
		if (rng_type=='pgh'):
			self.r_ng = self.r_pgh
		else:
			print("ERROR: Unkown R_nG \'{}\' used, please choose from:".format(rng_type))
			print("- Paranjape-Gordon-Hotchkiss (\'pgh\')")
			
	
	def display(self):
		print("{}, {}".format(self.label, self.params))
		
	def tinker(self, sigma, z):
	
		aa = self.params[0]
		a = self.params[1]
		b = self.params[2]
		c = self.params[3]
		
		aa_z = aa * pow((1.e0+z), -0.14e0)
		a_z = a * pow((1.e0+z), -0.06e0)
		alpha = exp(-(pow(0.75e0/(log(200.e0/0.75e0)), 1.2e0)))
		b_z = b * pow((1.e0+z), -alpha)

		f_t = aa_z * ( pow(sigma/b_z, -a_z) + 1.e0 ) * exp( -c/(sigma*sigma) )

		return f_t
		
	def st(self, sigma, z):
		
		aa = self.params[0]
		a = self.params[1]
		p = self.params[2]
		delta_c = self.params[3]
		
		return aa * sqrt(2.e0*a/pi) * (delta_c/sigma) * exp( (-a*delta_c*delta_c) / (2.e0*sigma*sigma) )*(1.e0 + pow( ( (sigma*sigma)/(a*delta_c*delta_c) ) , p ))
		
	def ps(self, sigma, z):
	
		delta_c = self.params[0]
	
		return sqrt(2.e0/pi) * (delta_c/sigma) * exp( (-delta_c*delta_c)/(2.e0*sigma*sigma) )
		
	def r_pgh(self, sigma, delta_c, f_nl):

		if (f_nl < 0):
	
			f_nl = abs(f_nl)

			delta_c = sqrt(0.837)*delta_c #Tinker / sph. collapse correction factor.

			nu = delta_c / sigma
			pgh_ng=0.e0
			e_1 = 3.e-4 * f_nl

			pgh_ng = sqrt(1.e0/( 1 + e_1*nu )) * exp( (nu*nu/2.e0) + (1/(e_1*e_1))*( e_1*nu - (1+e_1*nu)*log(1+e_1*nu) ) )

			return 1.e0/pgh_ng

		elif (f_nl == 0):
			return 1.e0
		else:
				
			delta_c = sqrt(0.837)*delta_c #Tinker / sph. collapse correction factor.

			nu = delta_c / sigma
			pgh_ng=0.e0
			e_1 = 3.e-4 * f_nl

			pgh_ng = sqrt(1.e0/( 1 + e_1*nu )) * exp( (nu*nu/2.e0) + (1/(e_1*e_1))*( e_1*nu - (1+e_1*nu)*log(1+e_1*nu) ) )

			return pgh_ng
