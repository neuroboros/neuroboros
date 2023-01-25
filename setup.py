from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='neuroboros',
    scripts=['bin/npls', 'bin/rmdirs'],
    ext_modules=cythonize("src/neuroboros/*/*.pyx"),
    include_dirs=[numpy.get_include()],
)
