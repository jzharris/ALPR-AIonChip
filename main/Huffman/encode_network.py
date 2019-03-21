import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
            encoded_obj = encode_np(val_np)

            # increment weight count by number of weights in layer
            weight_count += len(val_np.flatten())
            layer_count += 1

    print(">>>\t encoded a total of {} layers, and {} weights".format(layer_count, weight_count))


def encode_np(val_np):
    # gather statistics for each value in the np array
    freq = get_freqs(val_np)


def get_freqs(val_np):
    pass