from setuptools import setup, Extension
from Cython.Build import build_ext

import numpy
npy_include_dir = numpy.get_include()

ext_modules = [Extension("aco2d", ["pyacofwi2D.pyx", "fdtd2D_modelling.c",
                                    "cpml2D.c", "vdacoustic2D_modelling.c", "vdcpml2D.c"],
                         include_dirs = [npy_include_dir],
                         depends=["pyacofwi2D.h", "fdtd2D_modelling.h",
                                  "cpml2D.h", "cpml2D_kernels.h",
                                  "vdacoustic2D_modelling.h", "vdcpml2D.h",
                                  "vdcpml2D_kernels.h"],
                         language='c',
                         extra_compile_args=["-fopenmp"],
                         extra_link_args=["-fopenmp"],
                         libraries=["m"],
                         #extra_objects=["fdtd2D_modelling.o"]
                         )]

setup(name = 'acousticfwi2D',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
