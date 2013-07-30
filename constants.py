"""
constants (dictionary)

"""


const = {} # constants dictionary initialiser
convert = {} # conversion dictionary initialiser

const["G"] = 6.6726e-11   # Gravitational constant
const["c"] = 299792458.   # Speed of light (m s^-1)


convert['yr_s'] = 3.155815e7    # year in seconds
convert['msol_kg'] = 1.989e30   # solar mass in kg
convert['pc_m'] = 3.0856e16     # parsec in metres
convert['Mpc_m'] = 3.0856e22    # Mega parsec in metres
convert['Mpc_km'] = 3.0856e19   # Mega Parsec in kilometres
convert['H0'] = 3.2407793e-18   # Hubbles parameter (km /s /Mpc = /s)


# ------------------------------------------------------------------------
# constants of nature
# ------------------------------------------------------------------------
G = 6.6726e-11
c_light = 299792458. # m/s

# ------------------------------------------------------------------------
# unit conversion factors
# ------------------------------------------------------------------------
yr = 3.155815e7                                       # year in seconds
msol = 1.989e30                                       # solar mass in kg
pc = 3.0856e16                                        # parsec in metres
rhofactor = pow(yr,-2.e0)*(pow(1e6*pc, 3.e0))/msol
Hfactor = 1.e3*pc/yr
