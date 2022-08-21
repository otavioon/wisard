from os import urandom
from typing import List, Tuple

import numpy as np
from numba import jit
import tqdm

from .lut import LUT
from .bloom_filter import generate_h3_values, BloomFilter


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
def int_to_binary_list(value: int, size: int = 16):
    return [(value >> i) % 2 for i in range(size)]


# Implementes a single discriminator in the WiSARD model
# A discriminator is a collection of boolean LUTs with associated input sets
# During inference, the outputs of all LUTs are summed to produce a response
class Discriminator:
    # Constructor
    # Inputs:
    #  num_inputs:    The total number of inputs to the discriminator
    #  unit_inputs:   The number of boolean inputs to each LUT/filter in the discriminator
    #  unit_entries:  The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:   The number of hash functions for each filter.
    #  random_values: If provided, is used to set the random hash seeds for all filters. Otherwise, each filter generates its own seeds.
    def __init__(self,
                 num_inputs,
                 unit_inputs,
                 unit_entries,
                 unit_hashes,
                 random_values=None,
                 use_hashing=False):
        assert ((num_inputs / unit_inputs).is_integer())
        self.num_filters = num_inputs // unit_inputs
        self.unit_inputs = unit_inputs
        self.use_hashing = use_hashing
        if use_hashing:
            self.filters = [
                BloomFilter(unit_inputs, unit_entries, unit_hashes,
                            random_values) for i in range(self.num_filters)
            ]
        else:
            self.filters = [LUT(unit_inputs) for i in range(self.num_filters)]

    # Performs a training step (updating filter values)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters,
                                   -1)  # Divide the inputs between the filters
        for idx, inp in enumerate(filter_inputs):
            # inp_val = input_to_value(inp)
            self.filters[idx].add_member(inp)

    # Performs an inference to generate a response (number of filters which return True)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: The response of the discriminator to the input
    def predict(self, xv, soft_error_rate=0.0):
        filter_inputs = xv.reshape(self.num_filters,
                                   -1)  # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            # inp_val = input_to_value(inp)
            response += int(self.filters[idx].check_membership(
                inp, soft_error_rate))
        return response

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for f in self.filters:
            f.binarize()

    # Corrupts all filters; this process is reversible using "uncorrupt"
    # Assumes bleaching of 1 (i.e. "binarize" should have been called prior)
    def corrupt(self, persistent_error_rate):
        for f in self.filters:
            f.corrupt(persistent_error_rate)

    # Reverts data corrupted using the "corrupt" function
    def uncorrupt(self):
        for f in self.filters:
            f.uncorrupt()

    def mental_image(self, input_idxs) -> Tuple[np.ndarray, np.ndarray]:
        img_0s = np.zeros(self.num_filters * self.unit_inputs)
        img_1s = np.zeros(self.num_filters * self.unit_inputs)

        if self.use_hashing:
            raise NotImplementedError

        for i, f in enumerate(self.filters):
            indexes = input_idxs[self.unit_inputs * i:self.unit_inputs *
                                 (i + 1)]
            for address in range(f.data.size):
                bin_address = int_to_binary_list(address, size=self.unit_inputs)
                if f.data[address] != 0:
                    for k in range(self.unit_inputs):
                        if bin_address[k] != 0:
                            img_1s[indexes[k]] += f.data[address]
                        else:
                            img_0s[indexes[k]] += f.data[address]

        return img_0s, img_1s

    def max_bleach(self):
        return max(f.data.max() for f in self.filters)


# Top-level class for the WiSARD weightless neural network model
class WiSARD:
    # Constructor
    # Inputs:
    #  num_inputs:       The total number of inputs to the model
    #  num_classes:      The number of distinct possible outputs of the model; the number of classes in the dataset
    #  unit_inputs:      The number of boolean inputs to each LUT/filter in the model
    #  unit_entries:     The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:      The number of hash functions for each filter.
    #  input_idxs:       If provided, supplies the indices of the input which should be used; this allows inputs to be used multiple times or not at all.
    #  shared_rand_vals: If true, use the same random hash seeds for all filters in the model. There doesn't seem to be any reason not to do this.
    def __init__(self,
                 num_inputs: int,
                 num_classes: int,
                 unit_inputs: int = 1,
                 unit_entries: int = 1,
                 unit_hashes: int = 1,
                 input_idxs: List[int] = None,
                 shared_rand_vals=True,
                 randomize: bool = True):
        self.pad_zeros = (((num_inputs // unit_inputs) * unit_inputs) -
                          num_inputs) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        if input_idxs is None:
            self.input_order = np.arange(
                pad_inputs)  # Use each input exactly once
        else:
            self.input_order = input_idxs

        if randomize:
            np.random.shuffle(
                self.input_order)  # Randomize the ordering of the inputs

        random_values = generate_h3_values(
            unit_inputs, unit_entries, unit_hashes) if shared_rand_vals \
            else None  # Generate hash seeds, if desired

        self.discriminators = [
            Discriminator(self.input_order.size, unit_inputs, unit_entries,
                          unit_hashes, random_values)
            for i in range(num_classes)
        ]

    # Performs a training step (updating filter values) for all discriminators
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train_sample(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        self.discriminators[label].train(xv)

    # Performs an inference with the provided input
    # Passes the input through all discriminators, and returns the one or more with the maximal response
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: A vector containing the indices of the discriminators with maximal response
    def predict_sample(self, xv, soft_error_rate=0.0):
        if (soft_error_rate > 0.0):
            assert (self.discriminators[0].filters[0].bleach[...] == 1
                   )  # Unsupported for other values
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        responses = np.array(
            [d.predict(xv, soft_error_rate) for d in self.discriminators],
            dtype=int)
        max_response = responses.max()
        return np.where(responses == max_response)[0]

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for d in self.discriminators:
            d.binarize()

    # Corrupts all filters; this process is reversible using "uncorrupt"
    def corrupt(self, persistent_error_rate):
        assert (self.discriminators[0].filters[0].bleach[...] == 1
               )  # Unsupported for other values
        for d in self.discriminators:
            d.corrupt(persistent_error_rate)

    # Reverts data corrupted using the "corrupt" function
    def uncorrupt(self):
        assert (self.discriminators[0].filters[0].bleach[...] == 1
               )  # Unsupported for other values
        for d in self.discriminators:
            d.uncorrupt()

    def __do_extract_mental_image(self, discriminator_no):
        return (discriminator_no,
                self.discriminators[discriminator_no].mental_image(
                    self.input_order))

    def mental_images(self,
                      discriminators_no: List[int] = None,
                      mode: str = "normal",
                      workers: int = None):
        discriminators_no = discriminators_no or range(len(self.discriminators))
        return [
            self.discriminators[d_no].mental_image(self.input_order)
            for d_no in tqdm.tqdm(discriminators_no,
                                  desc="Extracting mental images")
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, use_tqdm: bool = True):
        it = range(len(X))
        if use_tqdm:
            it = tqdm.tqdm(it,
                           total=len(X),
                           desc="Training model",
                           leave=True,
                           position=0)
        for i in it:
            self.train_sample(X[i], y[i])

    def predict(self,
                X: np.ndarray,
                y: np.ndarray,
                use_tqdm: bool = True,
                bleach: int = 1):
        self.set_bleaching(bleach)
        if use_tqdm:
            X = tqdm.tqdm(X,
                          desc="Evaluating model",
                          total=len(X),
                          leave=True,
                          position=0)
        y_pred = [self.predict_sample(x) for x in X]
        return y_pred

    def max_bleach(self):
        return max(d.max_bleach() for d in self.discriminators)


def model_from_coded_mental_image(model, coded_images_0s, coded_images_1s):
    for d_no, (mental_img_0,
               mental_img_1) in enumerate(zip(coded_images_0s,
                                              coded_images_1s)):
        num_filters = model.discriminators[d_no].num_filters
        img_0 = mental_img_0[model.input_order].reshape(num_filters, -1)
        img_1 = mental_img_1[model.input_order].reshape(num_filters, -1)

        for f_no, ram_value in enumerate(img_1):
            original_ram_value = ram_value.copy()
            bin_addresses = np.asarray(int_to_binary_list(2**len(ram_value) -
                                                          1))
            for i in reversed(range(len(ram_value))):
                if ram_value[i] > 0:
                    model.discriminators[d_no].filters[f_no].add_member(
                        bin_addresses, ram_value[i])
                    for j in reversed(range(i)):
                        ram_value[j] -= ram_value[i]
                bin_addresses[i] = 0
            model.discriminators[d_no].filters[f_no].add_member(
                bin_addresses, img_0[f_no][0])
