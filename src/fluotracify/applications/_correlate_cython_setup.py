import setuptools
from Cython.Build import cythonize
import numpy as np


setuptools.setup(
    ext_modules=cythonize("correlate_cython.pyx",
                          compiler_directives={'language_level': '3'}),
    include_dirs=[np.get_include()]
)
