"""
(HOPEFULLY) USEFUL POWER SPECTRUM CLASS.
NOTE ALL UNITS ARE WRT h^-1

(C) IAN HARRISON
2012-
IAN.HARRISON@ASTRO.CF.AC.UK

"""

from numpy import *

class PowSpec:

	def __init__(self):
		self.sigma = self.sigma_wmap7fit
		self.dlnsigmadlnm = self.dlnsigmadlnm_wmap7fit
		self.label = "fitted to WMAP7->CAMB"
		self.clfile="wmap7_bao_h0_scalCls.dat"
		
	def display(self):
		print("Power spectrum {}".format(self.label))
		
	def sigma_wmap7fit(self, lnm):
		return exp(18.0058 - 1.47523*lnm + 0.0385449*lnm*lnm - 0.0000112539*pow(lnm,4) + (1.3719e-9)*pow(lnm,6))
		
	def dlnsigmadlnm_wmap7fit(self, lnm):
		return -1.47523 + 0.0770898*lnm - 0.0000450156*pow(lnm,3) + (8.23139e-9)*pow(lnm,5)
