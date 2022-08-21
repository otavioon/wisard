from collections import defaultdict

import numpy as np
from numba import jit
from .filter import Filter


# Non-hashed conventional LUT, as was used in the original WiSARD
class LUT(Filter):
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.bleach = 1
        self.d = defaultdict(int)

    def check_membership(self, xv, soft_error_rate):
        val = self.d.get(xv, 0)
        return val >= self.bleach

    def add_member(self, xv, inc_val: int = 1):
        self.d[xv] += 1

    def set_bleaching(self, bleach):
        self.bleach = bleach
