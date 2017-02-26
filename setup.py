from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("hw13", ["hw13.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("hw14", ["hw14.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]

setup(
  name = 'hw1',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
