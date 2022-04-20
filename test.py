import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
import pysimplednn as pysimplednn
import numpy as np

pysimplednn.simplednn(np.array([1, 2]), 10)