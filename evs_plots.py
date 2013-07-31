"""
Code for creating plots of example Generalised Extreme Value (GEV)
distributions.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.stats import genextreme as gev

# set up fonts for plotting
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=12)

x = np.linspace(-4,7,100) # to evaluate GEVs over

fig = plt.figure(1, figsize=(7, 3.25))
fig.clf()

ax1 = fig.add_subplot(121)
ax1.plot(x, gev.pdf(x, 0), '-r', label='$\gamma=0$')
ax1.plot(x, gev.pdf(x, 0.28), '-g', label='$\gamma=0.28$')
ax1.plot(x, gev.pdf(x, 0.56), '-b', label='$\gamma=0.56$')
ax1.set_xlim([-4,7])
ax1.set_xlabel('$x$')
ax1.set_ylabel('$G_{\mathrm{GEV}}(x)$')
ax1.legend(loc='upper right', frameon=False, handletextpad=0)
ax1.yaxis.set_ticks([0,0.1,0.2,0.3,0.4])

ax2 = fig.add_subplot(122)
ax2.plot(x, gev.pdf(x, 0), '-r', label='$\gamma=0$')
ax2.plot(x, gev.pdf(x, -0.26), '-g', label='$\gamma=-0.28$')
ax2.plot(x, gev.pdf(x, -0.56), '-b', label='$\gamma=-0.56$')
ax2.set_xlim([-4,7])
ax2.set_xlabel('$x$')
ax2.legend(loc='upper right', frameon=False, handletextpad=0)
ax2.yaxis.set_ticks([0,0.1,0.2,0.3,0.4])
ax2.yaxis.set_ticklabels([])

"""
ax3 = fig.add_subplot(133)
ax3.plot(x, gev.pdf(x, 0.5), '-r')
ax3.set_xlabel('$x$')
ax3.set_title('Weibull $\gamma=-0.3$')
ax3.yaxis.set_ticks([0,0.1,0.2,0.3,0.4])
ax3.yaxis.set_ticklabels([])
"""

plt.savefig('gev.eps', format='eps', bbox_inches='tight')
#plt.show()
