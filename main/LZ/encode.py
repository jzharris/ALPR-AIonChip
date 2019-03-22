import argparse
import os
import json
import tensorflow as tf

from encode_lz import encode_layers

##########################################################################################################
# run: python encode.py -c config_encode_yolo.json 2>&1 | tee logs.txt
##########################################################################################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

    # skip specific types of variables/layers
    white_regex = config['lempel-ziv']['white_regex']

    # checkpoint paths
    input_checkpoint = config['lempel-ziv']['input_checkpoint']
    encoded_name = config['lempel-ziv']['encoded_name']
    verbose = config['lempel-ziv']['verbose']

    # output paths
    parent_folder = config['convert']['convert_dir']
    output_folder = "converted_checkpoint"
    output_path = os.path.join(parent_folder, output_folder)

    if not os.path.exists(output_path):
        raise Exception("ERROR: converted checkpoint not found at {}".format(output_path))

    ###############################
    #   Load the model and encode
    ###############################

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(output_path, '{}.ckpt.meta'.format(input_checkpoint)))
        new_saver.restore(sess, tf.train.latest_checkpoint(output_path))

        encode_layers(sess, white_regex, verbose)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)