import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import heapq

from pprint import pprint


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __gt__(self, other):
        if (other == None):
            return -1
        if (not isinstance(other, HeapNode)):
            return -1
        return self.freq > other.freq


class LempelZivCoding:
    def __init__(self, val_np):
        self.val_np = val_np
        self.combinations = {}

    def encode_np(self):
        # gather statistics for each value in the np array
        codebook_size, encoded_size = self.make_codes()
        return codebook_size, encoded_size

    def make_codes(self):
        self.combinations = {}
        # init table to single-variable instances
        unique = np.unique(self.val_np)
        code_length = len(unique)
        for i in range(code_length):
            self.combinations[str(unique[i])] = i
        print(self.combinations)

        output_code = []
        s1 = self.val_np.ravel()
        p = ''
        c = ''
        p += str(s1[0])
        code = code_length
        for i in range(len(s1)):
            if i != len(s1) - 1:
                c += str(s1[i + 1])
            if (p + c) in self.combinations:
                p = p + c
            else:
                # print('{}\t{}\t{}\t{}'.format(p, self.combinations[p], p + c, code))
                output_code.append(self.combinations[p])
                self.combinations[p + c] = code
                code += 1
                p = c
            c = ''
        output_code.append(self.combinations[p])
        print(output_code)

        codebook_size = code_length # need to store the original unique elements
        encoded_size = len(output_code) * 8 # need to store 8-bit uints for all the codes

        print('>>> {} 32-bit floating point numbers needed for codebook'.format(codebook_size))
        print('>>> {} bits needed for encoded variables'.format(encoded_size))
        return codebook_size, encoded_size


def encode_layers(sess, white_regex=None, verbose=True):
    print('Encoding network...')

    codebook_sizes = []
    encoded_bits = []

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    weight_count = 0
    layer_count = 0
    for v in (tqdm(all_vars) if not verbose else all_vars):
        skip = False
        for regex in white_regex:
            if regex in v.name:
                skip = True
        if skip:
            if verbose:
                print('>>> skipping {}, part of whitelist'.format(v.name))
                sys.stdout.flush()
        else:
            if verbose:
                print('>>> encoding {}'.format(v.name))
                sys.stdout.flush()

            # perform the encoding for the layer
            val_np = sess.run(v)
            encoder = LempelZivCoding(val_np)
            codebook_size, encoded_size = encoder.encode_np()
            codebook_sizes.append(codebook_size)
            encoded_bits.append(encoded_size) # number of bits needed to represent this layer

            # increment weight count by number of weights in layer
            layer_weights = len(val_np.flatten())
            weight_count += layer_weights
            layer_count += 1

    print(">>>")
    print(">>> encoded a total of {} layers, and {} weights".format(layer_count, weight_count))
    print(">>> a total of {} codebooks containing {} 32-bit floating point numbers ({:.2f} KB) was created".
          format(len(codebook_sizes), sum(codebook_sizes), sum(codebook_sizes) * 32 / 8000))
    print(">>> a total of {:.2f} KB are required to store the encoded variables ({:.2f} KB per layer on average)".
          format(sum(encoded_bits)/8000, np.average(np.array(encoded_bits))/8000))
    original_kb = weight_count * 32 / 8000
    print(">>> original number of bits needed: {:.2f} KB".format(original_kb))
    new_kb = (sum(codebook_sizes) * 32 + sum(encoded_bits)) / 8000
    print(">>> new number of bits needed:      {:.2f} KB".format(new_kb))
    print(">>> compression ratio:              {:.4f}".format(original_kb / new_kb))
