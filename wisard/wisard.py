from os import urandom
from typing import List

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
    def __init__(self, num_inputs, unit_inputs, unit_entries,
                 unit_hashes, random_values=None, use_hashing=False):
        assert((num_inputs/unit_inputs).is_integer())
        self.num_filters = num_inputs // unit_inputs
        self.unit_inputs = unit_inputs
        self.use_hashing = use_hashing
        if use_hashing:
            self.filters = [
                BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values) 
                for i in range(self.num_filters)
            ]
        else:
            self.filters = [LUT(unit_inputs) for i in range(self.num_filters)]

    # Performs a training step (updating filter values)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1)  # Divide the inputs between the filters
        for idx, inp in enumerate(filter_inputs):
            # inp_val = input_to_value(inp)
            self.filters[idx].add_member(inp)

    # Performs an inference to generate a response (number of filters which return True)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: The response of the discriminator to the input
    def predict(self, xv, soft_error_rate=0.0):
        filter_inputs = xv.reshape(self.num_filters, -1)  # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            # inp_val = input_to_value(inp)
            response += int(self.filters[idx].check_membership(inp,soft_error_rate))
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

    def mental_image(self, input_idxs):
        img = np.zeros(self.num_filters*self.unit_inputs)

        if self.use_hashing:
            raise NotImplementedError

        for i, f in enumerate(self.filters):
            indexes = input_idxs[self.unit_inputs*i : self.unit_inputs*(i+1)]
            for address in range(f.data.size):
                bin_address = int_to_binary_list(address, size=self.unit_inputs)
                if f.data[address] != 0:
                    for k in range(self.unit_inputs):
                        if bin_address[k] != 0:
                            img[indexes[k]] += f.data[address]
        return img


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
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes, input_idxs=None, shared_rand_vals=True, randomize: bool = True):
        self.pad_zeros = (((num_inputs // unit_inputs) * unit_inputs) - num_inputs) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        if input_idxs is None:
            self.input_order = np.arange(pad_inputs) # Use each input exactly once
        else:
            self.input_order = input_idxs

        if randomize:
            np.random.seed(int.from_bytes(urandom(4), "little"))
            np.random.shuffle(self.input_order) # Randomize the ordering of the inputs

        random_values = generate_h3_values(
            unit_inputs, unit_entries, unit_hashes) if shared_rand_vals \
            else None  # Generate hash seeds, if desired

        self.discriminators = [
            Discriminator(
                self.input_order.size, unit_inputs, unit_entries,
                unit_hashes, random_values)
            for i in range(num_classes)
        ]

    # Performs a training step (updating filter values) for all discriminators
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        self.discriminators[label].train(xv)

    # Performs an inference with the provided input
    # Passes the input through all discriminators, and returns the one or more with the maximal response
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: A vector containing the indices of the discriminators with maximal response
    def predict(self, xv, soft_error_rate=0.0):
        if (soft_error_rate > 0.0):
            assert(self.discriminators[0].filters[0].bleach[...] == 1)  # Unsupported for other values
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order]  # Reorder input
        responses = np.array(
            [d.predict(xv, soft_error_rate) for d in self.discriminators], 
            dtype=int
        )
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
        assert(self.discriminators[0].filters[0].bleach[...] == 1)  # Unsupported for other values
        for d in self.discriminators:
            d.corrupt(persistent_error_rate)

    # Reverts data corrupted using the "corrupt" function
    def uncorrupt(self):
        assert(self.discriminators[0].filters[0].bleach[...] == 1)  # Unsupported for other values
        for d in self.discriminators:
            d.uncorrupt()

    def __do_extract_mental_image(self, discriminator_no):
        return {discriminator_no: self.discriminators[discriminator_no].mental_image(self.input_order)}

    def mental_images(self, discriminators_no: List[int] = None, mode: str = "normal", workers: int = None):
        discriminators_no = discriminators_no or range(len(self.discriminators))
        return [
            self.discriminators[d_no].mental_image(self.input_order)
            for d_no in tqdm.tqdm(discriminators_no, desc="Extracting mental images")
        ]
