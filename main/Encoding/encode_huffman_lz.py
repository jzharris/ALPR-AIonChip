import argparse
import os
import json
import tensorflow as tf

from encode_network import encode_lz, encode_huff

##########################################################################################################
# run: python encode_huffman_lz.py -c config_encode_yolo.json 2>&1 | tee logs_huffman_lz.txt
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
    black_list = config['black_list']

    # checkpoint paths
    input_checkpoint = config['input_checkpoint']
    verbose = config['verbose']

    # output paths
    output_path = config['checkpoint_dir']

    if not os.path.exists(output_path):
        raise Exception("ERROR: converted checkpoint not found at {}".format(output_path))

    ###############################
    #   Load the model and encode
    ###############################

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(os.path.join(output_path, '{}.ckpt.meta'.format(input_checkpoint)))
        new_saver.restore(sess, tf.train.latest_checkpoint(output_path))

        # first stage of encoding: Huffman
        codes = encode_huff(sess, white_regex=black_list, verbose=verbose) #TODO
        # print(codes)

        # second stage of encoding: Huffman
        encode_lz(sess, codes=codes, verbose=verbose) #TODO


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)