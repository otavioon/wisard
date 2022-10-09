import numpy as np
import numba.core.types
from numba.typed import Dict
from numba import jit
from .filter import Filter


# Non-hashed conventional LUT, as was used in the original WiSARD
class DictLUT(Filter):
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.bleach = 1
        self.min_bleach = None
        self.d = Dict.empty(
            key_type=numba.core.types.string,
            value_type=numba.core.types.int64
        )

    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, bleach, min_bleach, data):
        # address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        # address = "".join(["0" if x==0 else "1" for x in xv])
        address = str(xv[0])
        val = data.get(address, 0)
        min_bleach = min_bleach or val
        return val >= bleach and val <= min_bleach

    @staticmethod
    @jit(nopython=True)
    def __add_member(xv, data, inc_val):
        # address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        # address = "".join(["0" if x==0 else "1" for x in xv])
        address = str(xv[0])
        if address not in data:
            data[address] = inc_val
        else:
            data[address] += inc_val

    def check_membership(self, xv, soft_error_rate):
        return DictLUT.__check_membership(xv, self.bleach, self.min_bleach, self.d)
        # address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        # val = self.d.get(address, 0)
        # return val >= self.bleach

    def add_member(self, xv, inc_val: int = 1):
        DictLUT.__add_member(xv, self.d, np.int64(inc_val))
        # address = (xv.astype(np.int64) * 2**np.arange(xv.size)).sum()
        # self.d[address] += 1

    def set_bleaching(self, bleach, min_bleach=None):
        self.bleach = bleach
        self.min_bleach = min_bleach

    def __getstate__(self):
        state = {
            "num_inputs": self.num_inputs,
            "bleach": self.bleach,
            "d": dict(self.d)
        }
        return state

    def __setstate__(self, state):
        self.__dict__["num_inputs"] = state["num_inputs"]
        self.__dict__["bleach"] = state["bleach"]
        self.__dict__["d"] = Dict.empty(
                key_type=numba.core.types.string,
                value_type=numba.core.types.int64
            )
        self.d.update(state["d"])