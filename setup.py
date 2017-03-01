from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

### list all .pyx files

our_modules = [
    Extension("hw13", ["hw13.pyx"], language="c++", extra_compile_args=['-fopenmp','-O3'], extra_link_args=['-fopenmp']),
    Extension("hw14", ["hw14.pyx"], language="c++", extra_compile_args=['-fopenmp',], extra_link_args=['-fopenmp']),
    Extension("hw14opt", ["hw14.pyx"], language="c++", extra_compile_args=['-fopenmp','-O3'], extra_link_args=['-fopenmp']),
]

### apparently equivalent ways to do setup

#setup(name = 'hw1', ext_modules=cythonize(our_modules))
setup(name = 'hw1', cmdclass = {'build_ext': build_ext}, ext_modules = our_modules)
