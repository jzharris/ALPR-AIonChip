import os

# disable GPU - not needed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import keras
import json
import argparse
from preprocessing import parse_annotation
import numpy as np
import shutil

from frontend import YOLO
from prune_network import CustomAdam

######################################################################
# To run: python convert_checkpoint.py -c config_lp_seg_mobilenet.json
######################################################################

parent_folder = "./"
filename = "lp_seg_mobilenet_last_unpruned"
checkpoint_name = "pruned_post-train"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf

    # make a output directory to save the files to
    output_folder = "converted_checkpoint"
    output_path = os.path.join(parent_folder, output_folder)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    full_path = os.path.join(parent_folder, filename)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()

    grad_mask_consts = None

    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'],
                grad_mask_consts=grad_mask_consts)

    # set parameters necessary for loss function
    yolo.set_params(train_imgs=train_imgs,
                    valid_imgs=valid_imgs,
                    train_times=config['train']['train_times'],
                    valid_times=config['valid']['valid_times'],
                    batch_size=config['train']['batch_size'],
                    warmup_epochs=config['train']['warmup_epochs'],
                    object_scale=config['train']['object_scale'],
                    no_object_scale=config['train']['no_object_scale'],
                    coord_scale=config['train']['coord_scale'],
                    class_scale=config['train']['class_scale'],
                    debug=config['train']['debug'])

    # Load model, print endpoints
    # optimizer = CustomAdam(lr=config['train']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = keras.models.load_model("{}.h5".format(full_path), custom_objects={'custom_loss': yolo.custom_loss,
                                                                               'CustomAdam': CustomAdam})
    print('inputs')
    print(model.inputs)
    print('outputs:')
    print(model.outputs)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess = keras.backend.get_session()
    save_path = saver.save(sess, os.path.join(output_path, "{}.ckpt".format(checkpoint_name)))


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
