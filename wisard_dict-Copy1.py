#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import sys
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from wisard.encoders import ThermometerEncoder, encode_dataset
from wisard.wisard import WiSARD, model_from_coded_mental_image
from wisard.utils import untie, get_random_permutation, permute_dataset_bits
from wisard.optimize import find_best_bleach_bayesian, find_best_bleach_bin_search

from wisard.data import IrisDataset
from keras.datasets import mnist, fashion_mnist



# In[2]:


def sample_digit(target: int, X, y):
    return next((digit for (digit, label) in zip(X, y)
                 if int(label) == int(target))).reshape((28, 28))


def display_mnist_digits(X,
                         y,
                         figsize=(16, 8),
                         vmin: float = None,
                         vmax: float = None,
                         cmap: str = "gray"):
    fig, axs = plt.subplots(2, 5, figsize=figsize, constrained_layout=True)

    for i in range(2):
        for j in range(5):
            im = axs[i, j].imshow(
                sample_digit(target=5 * i + j, X=X, y=y),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            axs[i, j].axes.xaxis.set_visible(False)
            axs[i, j].axes.yaxis.set_visible(False)
            if vmin is None or vmax is None:
                fig.colorbar(im, ax=axs[i, j], shrink=0.6)
    if vmin is not None and vmax is not None:
        fig.colorbar(im, ax=axs[:, 4], location="right", shrink=0.6)
    plt.show()


def do_train_and_evaluate(x_train,
                          y_train,
                          x_test,
                          y_test,
                          tuple_size: int,
                          input_indexes: List[int] = None,
                          shuffle_indexes: bool = True,
                          bleach: Union[int, str] = "auto",
                          **kwargs):
    num_classes = len(np.unique(y_train))

    print(" ----- Training model ----- ")            

    if input_indexes is None:
        input_indexes = np.arange(x_train[0].size)
    if shuffle_indexes:
        np.random.shuffle(input_indexes)
    print(f"Using input_indexes: {input_indexes}")

    model = WiSARD(num_inputs=x_train[0].size,
                   num_classes=num_classes,
                   unit_inputs=tuple_size,
                   unit_entries=1,
                   unit_hashes=1,
                   input_idxs=input_indexes,
                   shared_rand_vals=False,
                   randomize=False,
                   use_dict=True)

    print("Fitting...")
    model.fit(x_train, y_train, use_tqdm=False)
    max_bleach = model.max_bleach()
    print(f"Max bleach is: {max_bleach}\n")

    print(" ----- Evaluating model ----- ")

    if isinstance(bleach, int):
        y_pred = model.predict(x_test, y_test, bleach=bleach, use_tqdm=True)
        y_pred, ties = untie(y_pred, use_tqdm=False)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
    elif bleach == "auto":
        bleach = find_best_bleach_bin_search(model,
                                           X=x_test,
                                           y=y_test,
                                           min_bleach=1,
                                           max_bleach=max_bleach//2,
                                           **kwargs)
    else:
        raise ValueError(f"Invalid value for bleach: '{bleach}'")

    return model, bleach


# In[3]:


shared_rand_vals = True  # not used...
input_size = 24
num_classes = 10
tuple_size = 24
unit_entries = 1  # Only used for BloomFilter
unit_hashes = 1  # Only used for BloomFilter
input_idxs = np.random.shuffle(np.arange(input_size))  # Order to select elements
# input_idxs = np.arange(input_size).reshape(thermometer.resolution, -1).T.ravel()
randomize = False  # Randomize selection order?


# ## Randomized inputs

# In[6]:

# print("---------- Randomization -------------")

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# permutation = np.arange(np.unpackbits(x_train[0]).size)
# permutation = np.random.permutation(permutation)
# permutation

# x_train = permute_dataset_bits(x_train, permutation)
# x_test  = permute_dataset_bits(x_test, permutation)

# # display_mnist_digits(x_train, y_train, vmin=0, vmax=255)


# # In[7]:


# thermometer = ThermometerEncoder(minimum=0, maximum=255, resolution=24)
# x_train = encode_dataset(thermometer, x_train)
# x_test = encode_dataset(thermometer, x_test)


# # In[9]:


# input_size = x_train[0].size

# model, bleach = do_train_and_evaluate(x_train,
#                                       y_train,
#                                       x_test,
#                                       y_test,
#                                       input_indexes=None,
#                                       tuple_size=24,
#                                       bleach="auto",
#                                       shuffle_indexes=False)
# print(f"Best bleach: {bleach}")


# ## Normal MNIST


print("---------- Non-randomized -------------")

# In[10]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# display_mnist_digits(x_train, y_train, vmin=0, vmax=255)

thermometer = ThermometerEncoder(minimum=0, maximum=255, resolution=24)
x_train = encode_dataset(thermometer, x_train)
x_test = encode_dataset(thermometer, x_test)


# In[11]:


input_size = x_train[0].size

model, bleach = do_train_and_evaluate(x_train,
                                      y_train,
                                      x_test,
                                      y_test,
                                      input_indexes=None,
                                      tuple_size=24,
                                      bleach="auto",
                                      shuffle_indexes=True)
print(f"Best bleach: {bleach}")

