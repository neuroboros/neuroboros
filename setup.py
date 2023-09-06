import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='neuroboros',
    scripts=['bin/npls', 'bin/rmdirs'],
    ext_modules=cythonize("src/neuroboros/*/*.pyx"),
    include_dirs=[numpy.get_include()],
)
