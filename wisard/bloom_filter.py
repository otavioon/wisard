import numpy as np
from numba import jit


# NOTE: Curently prefering the H3 hash function over the Dietzfelbinger hash function for hardware implementability reasons

# Computes the Dietzfelbinger integer-integer hashing function
# This is a simple, non-modulo hashing function which is both universal and obeys the uniform difference property
# Inputs:
#  x: A w-bit integer to be hashed
#  a: A vector of random, odd, positive w-bit integers < 2**w
#  b: A vector of random, non-negative integers < 2**(w-M)
#  w: The bit width of the input
#  M: The bit width of the output
# Returns: A vector of the hashed results for each value of a&b
@jit(nopython=True)
def dietzfelbinger_hash(x, a, b, w, M):
    lsb_mask = (1 << w) - 1
    mult_result = (a*x+b) & lsb_mask # (ax+b)[w-1:0]
    return mult_result >> (w - M) # (ax+b)[w-1:w-M]


# Generates random a and b vectors for the Dietzfelbinger hash function
# Inputs:
#  num_inputs:  The bit width of the input to the hash function
#  num_entries: The address space which should be spanned by the hash output, i.e. 2**(hash output bits). Must be a power of 2.
#  num_hashes:  The number of values for a and b to generate, equivalent to the number of distinct hashes which should be computed for an input
# Returns: The vectors a_values, b_values, representing the a and b values for the hash
def generate_dietzfelbinger_values(num_inputs, num_entries, num_hashes):
    assert(np.log2(num_entries).is_integer())
    a_values = 2 * np.random.randint(0, 1 << (num_inputs-1), num_hashes) + 1 # Odd integers < 2**num_inputs
    b_values = np.random.randint(0, 1 << (num_inputs-int(np.log2(num_entries))), num_hashes) # Odd integers < 2**num_inputs-log(num_entries)
    return a_values, b_value


# Computes hash functions within the H3 family of integer-integer hashing functions,
#  as described by Carter and Wegman in the paper "Universal Classes of Hash Functions"
# This function requires more unique parameters than the Dietzfelbinger function, but avoids arithmetic
# Inputs:
#  xv: A bitvector to be hashed to an integer
#  m: An array of arrays of length equivalent to the length of xv, with entries of size equivalent to the hash size
@jit(nopython=True)
def h3_hash(xv, m):
    # selected_entries = np.where(xv, m, 0)
    selected_entries = xv * m  # np.where is unsupported in Numba
    # reduction_result = np.bitwise_xor.reduce(selected_entries, axis=1)
    reduction_result = np.zeros(m.shape[0], dtype=np.int64)  # ".reduce" is unsupported in Numba
    for i in range(m.shape[1]):
        reduction_result ^= selected_entries[:, i]
    return reduction_result


# Generates a matrix of random values for use as m-arrays for H3 hash functions
def generate_h3_values(num_inputs, num_entries, num_hashes):
    assert(np.log2(num_entries).is_integer())
    shape = (num_hashes, num_inputs)
    values = np.random.randint(0, num_entries, shape)
    return values


# Implements a Bloom filter, a data structure for approximate set membership
# A Bloom filter can return one of two results: "possibly a member", and "definitely not a member"
# The risk of false positives increases with the number of elements stored relative to the underlying array size
# This implementation generalizes the basic concept to incorporate the notion of bleaching from WNN research
# With bleaching, we replace seen/not seen bits in the data structure with counters
# Elements can now be added to the data structure multiple times
# Our results now become "possibly added at least <b> times" and "definitely added fewer than <b> times"
# Increasing the bleaching threshold (the value of b) can improve accuracy
# Once the final bleaching threshold has been selected, this can be converted to a traditional Bloom filter
#  by evaluating the predicate "d[i] >= b" for all entries in the filter's data array
class BloomFilter:
    # Constructor
    # Inputs:
    #  num_inputs:    The bit width of the input to the filter (assumes the underlying inputs are single bits)
    #  num_entries:   The size of the underlying array for the filter. Must be a power of two. Increasing this reduces the risk of false positives.
    #  num_hashes:    The number of hash functions for the Bloom filter. This has a complex relation with false-positive rates
    #  random_values: Optionally specify values for hashing. If not provided, they will be generated.
    def __init__(self, num_inputs, num_entries, num_hashes, random_values=None):
        self.num_inputs, self.num_entries, self.num_hashes = num_inputs, num_entries, num_hashes
        if random_values is None:
            # self.a_values, self.b_values = generate_dietzfelbinger_values(num_inputs, num_entries, num_hashes)
            self.hash_values = generate_h3_values(num_inputs, num_entries, num_hashes)
        else:
            #self.a_values, self.b_values = random_values
            self.hash_values = random_values
        self.index_bits = int(np.log2(num_entries))
        self.data = np.zeros(num_entries, dtype=int)
        self.bleach = np.array(1, dtype=int)
        # self.persistent_error_vector = np.zeros(self.num_entries, dtype=int)

    # Implementation of the check_membership function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, hash_values, bleach, data, soft_error_rate):
        #hash_results = dietzfelbinger_hash(x, a_values, b_values, num_inputs, index_bits)
        hash_results = h3_hash(xv, hash_values)
        if soft_error_rate > 0.0:
            # Take XOR of (binary) results with random binary vector
            # Probability of a given entry in the random vector being 1 is given by soft_error_rate
            soft_error_vector = np.random.binomial(1, soft_error_rate, hash_results.size)
            hash_results ^= soft_error_vector
        least_entry = data[hash_results].min() # The most times the entry has possibly been previously seen
        return least_entry >= bleach

    # Check whether a value is a member of this filter (i.e. possibly seen at least b times)
    # Inputs:
    #  xv:              The bitvector to check the membership of
    #  soft_error_rate: The odds that a given bit read from the filter's LUT should be flipped; if nonzero, assumes bleaching threshold is 1
    # Returns: A boolean, which is true if xv has possibly been seen at least b times, and false if it definitely has not been
    def check_membership(self, xv, soft_error_rate=0.0):
        return BloomFilter.__check_membership(xv, self.hash_values, self.bleach, self.data, soft_error_rate)

    # Implementation of the add_member function
    # Coding in this style (as a static method) is necessary to use Numba for JIT compilation
    @staticmethod
    @jit(nopython=True)
    def __add_member(xv, hash_values, data):
        hash_results = h3_hash(xv, hash_values)
        least_entry = data[hash_results].min()  # The most times the entry has possibly been previously seen
        data[hash_results] = np.maximum(data[hash_results], least_entry+1) # Increment upper bound

    # Register a bitvector / increment its encountered count in the filter
    # Inputs:
    #  xv: The bitvector
    def add_member(self, xv):
        BloomFilter.__add_member(xv, self.hash_values, self.data)

    # Set the bleaching threshold, which is used to exclude members which have not possibly been seen at least b times
    # Inputs:
    #  bleach: The new value for b
    def set_bleaching(self, bleach):
        self.bleach[...] = bleach

    # Converts the filter into a "canonical" Bloom filter, with all entries either 0 or 1 and bleaching of 1
    # This operation will not impact the result of the check_membership function for any input
    # This operation is irreversible, so shouldn't be done until the optimal bleaching value has been selected
    def binarize(self):
        self.data = (self.data >= self.bleach).astype(int)
        self.set_bleaching(1)

    # Corrupts bits in the filter with probability given by persistent_corruption_rate
    # Assumes bleaching of 1 (i.e. "binarize" should have been called prior)
    # Data can be corrupted multiple times; self.persistent_error_vector tracks which bits to flip to restore original state
    def corrupt(self, persistent_error_rate):
        if (not hasattr(self, "persistent_error_vector")) or (self.persistent_error_vector is None):
            self.persistent_error_vector = np.zeros(self.num_entries, dtype=int)
        else:
            assert(0)
        error_vector = np.random.binomial(1, persistent_error_rate, self.num_entries)
        self.data ^= error_vector
        self.persistent_error_vector ^= error_vector

    # Reverts data corrupted using the "corrupt" function
    # No effect if data is not currently corrupted
    def uncorrupt(self):
        self.data ^= self.persistent_error_vector
        del self.persistent_error_vector

