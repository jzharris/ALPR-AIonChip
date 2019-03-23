import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math
import heapq
from pprint import pprint

########################################################################################################################
# printing stats from compression

def first_encoding_stats(layer_count, weight_count, codebook_sizes, code_bits, first_encoder="huff"):
    if first_encoder != "huff":
        first_encoder = "lziv"

    print(">>>")
    print(">>> encoded a total of {} layers, and {} weights".format(layer_count, weight_count))
    codebook_count = len(codebook_sizes)
    codes_count = sum(codebook_sizes)
    codebook_size = codes_count * 32 / 8000
    print(">>> a total of {} codebooks containing {} values ({:.2f} KB) was created".format(codebook_count,
                                                                                            codes_count,
                                                                                            codebook_size))

    quant_vals_kb = (256 * layer_count) * 32 / 8000  # the 256 32-bit floating point values available per layer
    quant_weights_kb = weight_count * 8 / 8000  # the amount of space it takes to represent the quantized weights
    original_kb = quant_vals_kb + quant_weights_kb
    print(">>>")
    print(">>>                       quantized values:   {:.2f} KB".format(quant_vals_kb))
    print(">>>      the quantized code representation: + {:.2f} KB".format(quant_weights_kb))
    print(">>> bits needed to store quantized network: = {:.2f} KB".format(original_kb))

    codebook_kb = sum(codebook_sizes) * 32 / 8000
    codes_kb = sum(code_bits) / 8000
    new_kb = codebook_kb + codes_kb

    print(">>>")
    print(">>>                       bits in codebook:   {:.2f} KB".format(codebook_kb))
    print(">>>          the {} codes representation: + {:.2f} KB".format(first_encoder, codes_kb))
    print(">>>      bits needed to store {} network: = {:.2f} KB".format(first_encoder, new_kb))
    print(">>>")

    compression_ratio = original_kb / new_kb
    print(">>>                      compression ratio:   {:.4f}".format(compression_ratio))
    print(">>>")

    return codebook_kb, codes_kb, original_kb


def second_encoding_stats(codebook_kb, codes_kb, original_kb, prev_codes, prev_codes_count, prev_codebook_sizes, code_bits, first_encoder="huff"):
    if first_encoder == "huff":
        second_encoder = "lziv"
    else:
        first_encoder = "lziv"
        second_encoder = "huff"

    print(">>>")
    prevbooks = len(prev_codes.keys())
    print(">>> encoded a total of {0} {1} codebooks, and {2} {1} codes".format(prevbooks, first_encoder, prev_codes_count))
    codebook_count = len(prev_codebook_sizes)
    prev_codes_count = sum(prev_codebook_sizes)
    codebook_size = prev_codes_count * 32 / 8000
    print(">>> a total of {} codebooks containing {} values ({:.2f} KB) was created".format(codebook_count,
                                                                                            prev_codes_count,
                                                                                            codebook_size))

    prev_kb = codebook_kb + codes_kb

    print(">>>")
    print(">>>                  bits in {} codebook:   {:.2f} KB".format(first_encoder, codebook_kb))
    print(">>>          the {} codes representation: + {:.2f} KB".format(first_encoder, codes_kb))
    print(">>>      bits needed to store {} network: = {:.2f} KB".format(first_encoder, prev_kb))

    codebook_kb = sum(prev_codebook_sizes) * 32 / 8000
    codes_kb = sum(code_bits) / 8000
    new_kb = codebook_kb + codes_kb

    print(">>>")
    print(">>>                  bits in {} codebook:   {:.2f} KB".format(second_encoder, codebook_kb))
    print(">>>          the {} codes representation: + {:.2f} KB".format(second_encoder, codes_kb))
    print(">>>      bits needed to store {} network: = {:.2f} KB".format(second_encoder, new_kb))
    print(">>>")

    compression_ratio = prev_kb / new_kb
    print(">>>                      compression ratio:   {:.4f}".format(compression_ratio))
    print(">>>")

    print("Determining overall compression ratio...")
    print(">>>")
    print(">>> bits needed to store quantized network:   {:.2f} KB".format(original_kb))
    print(">>>      bits needed to store {} network:   {:.2f} KB".format(second_encoder, new_kb))
    compression_ratio = original_kb / new_kb
    print(">>>                      compression ratio:   {:.4f}".format(compression_ratio))
    print(">>>")

########################################################################################################################
# huffman encoding

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
    def __init__(self, val_np, verbose=0):
        self.val_np = val_np
        self.verbose = verbose
        self.heap = []
        self.dictionary = {}
        self.codes = []

    def encode_np(self):
        # gather statistics for each value in the np array
        freqs_d = self.get_freqs()
        self.make_heap(freqs_d)
        self.merge_nodes()
        self.make_dict()
        codebook_size, encoded_size = self.make_stats()
        self.generate_codes()
        return codebook_size, encoded_size

    def get_freqs(self):
        unique, freqs = np.unique(self.val_np, return_counts=True)
        # freqs = freqs / len(self.val_np.flatten())

        # return unique, freqs
        freqs_d = dict(zip(unique, freqs))
        if(self.verbose >= 2):
            pprint(freqs_d)
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
            self.dictionary[root.char] = current_code
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_dict(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def make_stats(self, dtype='float'):
        if self.verbose >= 2:
            pprint(self.dictionary)

        # count how many floats needed for codebook
        codebook_size = len(self.dictionary.keys())

        # count the number of bits now needed to store variables
        encoded_size = 0
        for val in self.val_np.flatten():
            encoded_size += len(self.dictionary[val])

        if self.verbose >= 1:
            print('>>> {} {} numbers needed for codebook'.format(codebook_size,
                                                                 '32-bit floating point' if dtype=='float' else 'uint8'))
            print('>>> {} bits needed for encoded variables'.format(encoded_size))

        return codebook_size, encoded_size

    def generate_codes(self):
        # generate the codes needed for the codebook
        for val in self.val_np.flatten():
            self.codes.append(int(self.dictionary[val], 2))

        if self.verbose >= 2:
            print(self.codes)


def encode_huff(sess, codes=None, codebook_kb=None, codes_kb=None, original_kb=None, white_regex=None, verbose=2):
    if codes is None:
        print('Huffman encoding network...')
    else:
        print('Huffman encoding LZ codes...')

    if white_regex is None:
        white_regex = []

    if codes is None:
        codes = {}
        codebook_sizes = []
        code_bits = []
        weight_count = 0
        layer_count = 0
        # codes_bits = []

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        for v in (tqdm(all_vars) if verbose == -1 else all_vars):
            skip = False
            for regex in white_regex:
                if regex in v.name:
                    skip = True
            if skip:
                if verbose > 0:
                    print('>>> skipping {}, part of whitelist'.format(v.name))
                    sys.stdout.flush()
            else:
                if verbose > 0:
                    print('>>> encoding {}'.format(v.name))
                    sys.stdout.flush()

                # perform the encoding for the layer
                val_np = sess.run(v)
                encoder = HuffmanCoding(val_np, verbose=verbose)
                codebook_size, encoded_size = encoder.encode_np()
                codebook_sizes.append(codebook_size) # number of variables in the codebook for this layer
                code_bits.append(encoded_size) # number of bits needed to represent this layer
                codes[v.name] = encoder.codes

                # increment weight count by number of weights in layer
                layer_weights = len(val_np.flatten())
                weight_count += layer_weights
                layer_count += 1

        codebook_kb, codes_kb, original_kb = first_encoding_stats(layer_count, weight_count, codebook_sizes, code_bits, first_encoder="huff")
        return codebook_kb, codes_kb, original_kb, codes  # can be used to chain encoders

    else: # assuming previous compression is from LZ
        codebook_sizes = []
        codes_count = 0
        code_bits = []

        for v_name in (tqdm(codes.keys()) if verbose == -1 else codes.keys()):
            codes_count += len(codes[v_name])
            encoder = HuffmanCoding(np.array(codes[v_name]), verbose=verbose)
            codebook_size, encoded_size = encoder.encode_np()
            codebook_sizes.append(codebook_size) # number of variables in the codebook for this layer
            code_bits.append(encoded_size) # number of bits needed to represent this layer

        second_encoding_stats(codebook_kb, codes_kb, original_kb, codes, codes_count, codebook_sizes, code_bits, first_encoder="lziv")

########################################################################################################################
# lziv encoding

class LempelZivCoding:
    def __init__(self, val_np, verbose=0):
        self.val_np = val_np
        self.verbose = verbose
        self.combinations = {}
        self.codes = None

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
        if(self.verbose >= 2):
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

        if code > 2**32: # assuming 32-bit numbers
            raise Exception('code is: {} which is > 2^23'.format(code))

        output_code.append(self.combinations[p])
        if self.verbose >= 2:
            print('generated code dictionary:')
            print(output_code)
        self.codes = output_code

        codebook_size = code_length # need to store the original unique elements
        # encoded_size = len(output_code) * 32 # need to store uint32 variables for all the codes
        encoded_size = len(output_code) * math.ceil(math.log(max(output_code), 2)) # store only the variable size we need per layer

        if self.verbose >= 1:
            print('>>> {} 32-bit floating point numbers needed for codebook'.format(codebook_size))
            print('>>> {} bits needed for encoded variables'.format(encoded_size))
        return codebook_size, encoded_size


def encode_lziv(sess, codes=None, codebook_kb=None, codes_kb=None, original_kb=None, white_regex=None, verbose=2):
    if codes is None:
        print('LZ encoding network...')
    else:
        print('LZ encoding Huffman codes...')

    if white_regex is None:
        white_regex = []

    if codes is None:
        codes = {}
        codebook_sizes = []
        code_bits = []

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        weight_count = 0
        layer_count = 0
        for v in (tqdm(all_vars) if verbose == -1 else all_vars):
            skip = False
            for regex in white_regex:
                if regex in v.name:
                    skip = True
            if skip:
                if verbose > 0:
                    print('>>> skipping {}, part of whitelist'.format(v.name))
                    sys.stdout.flush()
            else:
                if verbose > 0:
                    print('>>> encoding {}'.format(v.name))
                    sys.stdout.flush()

                # perform the encoding for the layer
                val_np = sess.run(v)
                encoder = LempelZivCoding(val_np, verbose=verbose)
                codebook_size, encoded_size = encoder.encode_np()
                codebook_sizes.append(codebook_size) # number of variables in the codebook for this layer
                code_bits.append(encoded_size) # number of bits needed to represent this layer
                codes[v.name] = encoder.codes

                # increment weight count by number of weights in layer
                layer_weights = len(val_np.flatten())
                weight_count += layer_weights
                layer_count += 1

        codebook_kb, codes_kb, original_kb = first_encoding_stats(layer_count, weight_count, codebook_sizes, code_bits, first_encoder="lziv")
        return codebook_kb, codes_kb, original_kb, codes  # can be used to chain encoders

    else: # assuming previous compression is from Huffman
        codebook_sizes = []
        # encoded_bits = []
        codes_count = 0
        code_bits = []

        for v_name in (tqdm(codes.keys()) if verbose == -1 else codes.keys()):
            codes_count += len(codes[v_name])
            encoder = LempelZivCoding(np.array(codes[v_name]), verbose=verbose)
            codebook_size, encoded_size = encoder.encode_np()
            codebook_sizes.append(codebook_size) # number of variables in the codebook for this layer
            # encoded_bits.append(encoded_size) # number of bits needed to represent this layer

            # calculate the variable sizes of the codes in this layer
            # codes_bits.append(len(codes[v_name]) * math.ceil(math.log(max(codes[v_name]), 2))) # store only the variable size we need per layer
            code_bits.append(sum([x.bit_length() for x in codes[v_name]]))

        second_encoding_stats(codebook_kb, codes_kb, original_kb, codes, codes_count, codebook_sizes, code_bits, first_encoder="huff")
