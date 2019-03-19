#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import sys
import json



import tensorflow as tf
import numpy as np
import os
import sys

from tqdm import tqdm

# one method to do quantization
def strategy_1(v_numpy):
    # 8 bit int can represent for 256 levels
    NUM_OF_LEVEL = 256
    NUM_OF_LEVEL -= 1 # Remove one, save this spot for [0]

    v_numpy_shape = v_numpy.shape
    v_numpy_flatten = v_numpy.flatten()
    min_np = v_numpy.min()
    max_np = v_numpy.max()
    gap = (max_np - min_np) / (NUM_OF_LEVEL - 1)
    levels = np.asarray([(min_np + x * gap) for x in range(NUM_OF_LEVEL)] + [0]) # add 0, to pass pruned weights
    for i in range(len(v_numpy_flatten)):
        v_numpy_flatten[i] = min(levels, key=lambda x: abs(x - v_numpy_flatten[i]))
    v_numpy = v_numpy_flatten.reshape(v_numpy_shape)

    return v_numpy


def quantize_one_variable(sess, v):
    v_numpy = sess.run(v)
    new_v = strategy_1(v_numpy)
    assign_op = v.assign(new_v)
    return assign_op


def quantize_layers(sess, white_regex=None, verbose=True):
    print('Quantizing network...')
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    update_operation = []
    quantized_count = 0
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
                print('>>> quantizing {}'.format(v.name))
                sys.stdout.flush()
            update_operation.append(quantize_one_variable(sess, v))

            # increment weight count by number of weights in layer
            val_np = sess.run(v)
            quantized_count += len(val_np.flatten())
    _ = sess.run(update_operation)

    print(">>>\t quantized a total of {} weights".format(quantized_count))
