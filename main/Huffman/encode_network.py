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


class HuffmanCoding:
    def __init__(self, val_np):
        self.val_np = val_np
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    def encode_np(self):
        # gather statistics for each value in the np array
        freqs_d = self.get_freqs()
        self.make_heap(freqs_d)
        self.merge_nodes()
        self.make_codes()
        codebook_size, encoded_size = self.make_stats()
        return codebook_size, encoded_size

    def get_freqs(self):
        unique, freqs = np.unique(self.val_np, return_counts=True)
        # freqs = freqs / len(self.val_np.flatten())

        # return unique, freqs
        freqs_d = dict(zip(unique, freqs))
        # print(freqs_d)
        return freqs_d

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)
        pprint(self.codes)

    def make_stats(self):
        codes = self.codes

        # count how many floats needed for codebook
        codebook_size = len(codes.keys())

        # count the number of bits now needed to store variables
        encoded_size = 0
        for val in self.val_np.flatten():
            encoded_size += len(codes[val])

        # # count the largest bit size needed for encoded variables
        # longest_str = None
        # for key in codes.keys():
        #     if longest_str is None or len(codes[key]) > len(longest_str):
        #         longest_str = codes[key]
        # encoded_size = len(longest_str)

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
            encoder = HuffmanCoding(val_np)
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
