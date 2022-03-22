from project.compress import compress
from project.reconstruct import reconstruct
from project.reid import reid
from torch import nn

"""
A simple demo of the running environment of NAIC2022
Note that the extract function is not called since it takes much time
"""

compress('64')
reconstruct('64')
compress('128')
reconstruct('128')

compress('256')
reconstruct('256')

reid('64')
reid('128')
reid('256')
