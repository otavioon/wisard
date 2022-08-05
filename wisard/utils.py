from typing import List

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
