import numpy as np
from numba import jit


# Non-hashed conventional LUT, as was used in the original WiSARD
class LUT:

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.data = np.zeros(2**num_inputs, dtype=int)
        self.bleach = np.array(1, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, bleach, data):
        address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        result = data[address]
        return result >= bleach

    def check_membership(self, xv, soft_error_rate):
        assert (soft_error_rate == 0.0)  # NYI
        return LUT.__check_membership(xv, self.bleach, self.data)

    @staticmethod
    @jit(nopython=True)
    def __add_member(xv, data, inc_val):
        address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        data[address] += inc_val

    def add_member(self, xv, inc_val: int = 1):
        LUT.__add_member(xv, self.data, inc_val)

    def set_bleaching(self, bleach):
        self.bleach[...] = bleach

    def binarize(self):
        self.data = (self.data >= self.bleach).astype(int)
        self.set_bleaching(1)