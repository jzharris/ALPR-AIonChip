#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import sys
import json



import tensorflow as tf
import numpy as np
import os

# 8 bit int can represent for 256 levels
NUM_OF_LEVEL = 256
# TRAIN_DIR = '/media/qiujing/e5439522-63c4-4b7c-a968-fefee6a3d960/omead/aiOnChip/AIonChip_HOZ/main/LP_segmentation/pruned_models/converted_checkpoint'
# META = 'lp_seg_mobilenet_pruned_post-train_it25.ckpt.meta'
# CKPT = 'lp_seg_mobilenet_pruned_post-train_it25.ckpt'

##########################################################################################################
# run: python quantize.py -c config_lp_seg_mobilenet_quant.json 2>&1 | tee quant_models/logs.txt
##########################################################################################################

# one method to do quantization
def strategy_1(v_numpy):
    v_numpy_shape = v_numpy.shape
    v_numpy_flatten = v_numpy.flatten()
    min_np = v_numpy.min()
    max_np = v_numpy.max()
    gap = (max_np - min_np) / (NUM_OF_LEVEL - 1)
    levels = np.asarray([(min_np + x * gap) for x in range(NUM_OF_LEVEL)])
    for i in range(len(v_numpy_flatten)):
        v_numpy_flatten[i] = min(levels, key=lambda x : abs(x-v_numpy_flatten[i]))
    v_numpy = v_numpy_flatten.reshape(v_numpy_shape)
    return v_numpy


def quantize_one_variable(sess, v):
    v_numpy = sess.run(v)
    new_v = strategy_1(v_numpy)
    assign_op = v.assign(new_v)
    return assign_op


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_dir = config['quant']['train_dir']
    meta = config['quant']['meta']
    ckpt = config['quant']['ckpt']
    out = config['quant']['quantized_model']
    out_dir = config['quant']['quant_dir']

    # pretrained_weights = config['quant']['pretrained_weights']
    # skip_first_train = config['train']['skip_first_train']


    with tf.Session() as sess:
        # load the checkpoint
        saver = tf.train.import_meta_graph(os.path.join(train_dir, meta))
        saver.restore(sess, os.path.join(train_dir, ckpt))
        graph = tf.get_default_graph()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        update_operation = []
        for v in all_vars:
            # quantize only network variables
            if ('fcn' in v.name) and ('Adam' not in v.name):
                update_operation.append(quantize_one_variable(sess, v))
        _ = sess.run(update_operation)
        # save the qunatized checkpoint
        checkpoint_path = os.path.join(out_dir,out )
        saver.save(sess, checkpoint_path)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)