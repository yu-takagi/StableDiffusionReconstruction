"""
BdPy: Brain decoding toolbox for Python

Developed by Kamitani Lab, Kyoto Univ. and ATR
"""


# `import bdpy` implicitly imports class `BData` (in package `bdata`) and
# package `util`.
from .bdata import BData
from .bdata import vstack, metadata_equal
from .util import *
