from setuptools import setup, find_packages, Extension
import os

# Stuff used to build the PyCosmo.EH._power module:
power_module = Extension('EH._power',
                         sources=[os.path.join('EH','power_wrap.c'),
                                  os.path.join('EH','power.c')]
                         )


tf_fit_module = Extension('EH._tf_fit',
                         sources=[os.path.join('EH','tf_fit_wrap.c'),
                                  os.path.join('EH','tf_fit.c')]
                         )


packages = find_packages()

setup(
    name = "PyCosmo",
    version = "1.0",
    packages = packages,
    install_requires = ['numpy', 'scipy',],

    ext_modules = [power_module, tf_fit_module],
)
