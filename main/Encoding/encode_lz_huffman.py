import argparse
import os
import json
import tensorflow as tf

from encode_network import encode_lz, encode_huff

##########################################################################################################
# run: python encode_lz_huffman.py -c config_encode_yolo.json 2>&1 | tee logs_lz_huffman.txt
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
    white_regex = config['lz_huff']['white_regex']

    # checkpoint paths
    input_checkpoint = config['lz_huff']['input_checkpoint']
    encoded_name = config['lz_huff']['encoded_name']
    verbose = config['lz_huff']['verbose']

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

        # first stage of encoding: LZ
        codes = encode_lz(sess, white_regex, verbose)
        # print(codes)

        # second stage of encoding: Huffman
        encode_huff(sess, codes=codes)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)