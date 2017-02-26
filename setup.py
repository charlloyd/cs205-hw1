from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
  name = 'hw1',
  ext_modules=[
    Extension("hw13", ["hw13.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    Extension("hw14", ["hw14.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']),
    ],
  cmdclass = {'build_ext': build_ext}
)
