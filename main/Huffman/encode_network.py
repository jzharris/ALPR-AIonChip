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
        return self.codes

    def get_freqs(self):
        unique, freqs = np.unique(self.val_np, return_counts=True)
        # freqs = freqs / len(self.val_np.flatten())

        # return unique, freqs
        freqs_d = dict(zip(unique, freqs))
        print(freqs_d)
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


def encode_layers(sess, white_regex=None, verbose=True):
    print('Encoding network...')
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
            codes = encoder.encode_np()
            pprint(codes)
            exit(0)

            # increment weight count by number of weights in layer
            weight_count += len(val_np.flatten())
            layer_count += 1

    print(">>>\t encoded a total of {} layers, and {} weights".format(layer_count, weight_count))
