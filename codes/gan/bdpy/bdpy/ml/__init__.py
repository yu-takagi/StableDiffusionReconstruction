"""
BdPy machine learning package

This package is a part of BdPy
"""


from .learning import Classification, CrossValidation, ModelTraining, ModelTest
from .crossvalidation import make_cvindex, make_crossvalidationindex, make_cvindex_generator
from .crossvalidation import cvindex_groupwise
from .ensemble import *
from .regress import *
from .searchlight import *
