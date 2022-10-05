from typing import List
from numba import jit

import tqdm
import numpy as np


def untie(y_pred, use_tqdm: bool = True):
    final_pred = []
    ties = 0
    if use_tqdm:
        y_pred = tqdm.tqdm(y_pred, desc="Untieing")

    for pred in y_pred:
        if len(pred) > 1:
            ties += 1
            # final_pred.append(pred[random.randint(0, len(pred) - 1)])
            final_pred.append(pred[0])
        else:
            final_pred.append(pred[0])

    return np.array(final_pred), ties


def permute_dataset_bits(X: np.ndarray, permutation: List[int], use_tqdm: bool = True):
    X = X.astype(np.uint8)
    values = []
    if use_tqdm:
        X = tqdm.tqdm(X)
    for x in X:
        coded = np.unpackbits(x)[permutation]
        value = np.packbits(coded)
        values.append(value)
    return np.array(values)


def get_random_permutation(size: int):
    permutation = np.arange(np.unpackbits(size))
    permutation = np.random.permutation(permutation)
    return permutation


# Converts a vector of booleans to an unsigned integer
#  i.e. (2**0 * xv[0]) + (2**1 * xv[1]) + ... + (2**n * xv[n])
# Inputs:
#  xv: The boolean vector to be converted
# Returns: The unsigned integer representation of xv
@jit(nopython=True, inline='always')
def input_to_value(xv: np.ndarray):
    result = 0
    for i in range(xv.size):
        result += xv[i] << i
    return result


@jit(nopython=True, inline='always')
def int_to_binary_list(value: int, size: int):
    return [(value >> i) % 2 for i in range(size)]
