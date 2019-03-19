#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json

from quantize_network import quantize_layers
from prune_network import check_pruned_weights, get_mask_consts
import keras.backend as K

##########################################################################################################
# run: python train_quantize.py -c config_lp_seg_mobilenet_quant.json 2>&1 | tee quant_models/logs.txt
##########################################################################################################

# skip specific types of variables/layers
white_regex = ['CustomAdam', 'loss', 'training',
               'moving_mean', 'moving_variance', 'DetectionLayer']

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    skip_first_train = config['train']['skip_first_train']

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        print('Using validation set found in {}'.format(config['valid']['valid_annot_folder']))
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        print('No validation set found, splitting 20% of train set to use as validation instead')
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    print('Training shape:   {}'.format(len(train_imgs)))
    print('Validation shape: {}'.format(len(valid_imgs)))

    # training parameters
    iterations = config['train']['train_times']

    # parent save directory:
    quant_dir = config['quant']['quant_dir']
    if not os.path.isdir(quant_dir):
        os.mkdir(quant_dir)
    # find a unique folder to save to
    i = 1
    save_dir = "mobilenet_{}it_q_{}/".format(iterations, i)
    while os.path.isdir(os.path.join(quant_dir, save_dir)) and \
            (len(os.listdir(os.path.join(quant_dir, save_dir))) > 0):
        i += 1
        save_dir = "mobilenet_{}it_q_{}/".format(iterations, i)
    # folder to save things in this time:
    save_path = os.path.join(quant_dir, save_dir)
    if not os.path.isdir(save_path):  # dummy check
        os.mkdir(save_path)

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

    ####################################################################################################################
    for it in range(0, iterations):
        print("it {}".format(it))

        quant_weights_path_prev = os.path.join(save_path, config['quant']['weights_name'] + "_it{}.h5".format(it - 1))
        quant_weights_path_curr = os.path.join(save_path, config['quant']['weights_name'] + "_it{}.h5".format(it))

        ###############################
        #   Load the pretrained weights (if any)
        ###############################

        if it > 0 and os.path.exists(quant_weights_path_prev):
            print("Loading quantized weights from", quant_weights_path_prev)
            yolo.load_weights(quant_weights_path_prev)

            # check to make sure the weights are pruned
            sess = K.get_session()
            check_pruned_weights(sess, grad_mask_consts, 0, 0)

            #TODO: check to make sure weights have been quantized
        elif os.path.exists(config['train']['pretrained_weights']) and it == 0:
            print("Loading pre-trained weights from", config['train']['pretrained_weights'])
            yolo.load_weights(config['train']['pretrained_weights'])

            # get mask, if one exists
            # NOTE: expects a pruned checkpoint (pruned.h5, not pruned_post-train.h5)
            sess = K.get_session()
            new_mask = get_mask_consts(sess, [], white_regex, verbose=False)
            if new_mask != {}:
                grad_mask_consts = new_mask
                check_pruned_weights(sess, grad_mask_consts, 0, 0)

                print('Evaluating masked network, before train step:')
                yolo.validate(train_imgs=train_imgs,
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
        else:
            print("No pre-trained weights were loaded from {}".format(config['train']['pretrained_weights']))

        ###############################
        #   Start the training process
        ###############################

        if it > 0 or not skip_first_train:
            yolo.train(train_imgs=train_imgs,
                       valid_imgs=valid_imgs,
                       train_times=config['train']['train_times'],
                       valid_times=config['valid']['valid_times'],
                       nb_epochs=config['train']['nb_epochs'],
                       learning_rate=config['train']['learning_rate'],
                       batch_size=config['train']['batch_size'],
                       warmup_epochs=config['train']['warmup_epochs'],
                       object_scale=config['train']['object_scale'],
                       no_object_scale=config['train']['no_object_scale'],
                       coord_scale=config['train']['coord_scale'],
                       class_scale=config['train']['class_scale'],
                       saved_weights_name=os.path.join(save_path,
                                                       config['train']['saved_weights_name'] + "_it{}.h5".format(it)),
                       debug=config['train']['debug'],
                       verbose=True)
        else:
            # perform evaluation in either case (usually performed at end of training setp)
            print('Evaluating pre-trained network')
            yolo.validate(train_imgs=train_imgs,
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

        ###############################
        #   Quantize the layers
        ###############################

        sess = K.get_session()

        if it > 0:
            # show the weights just after training
            check_pruned_weights(sess, grad_mask_consts, 0, 0)

        # TODO: show that the weights are now unquantized

        # quantize the weights
        quantize_layers(sess, white_regex, verbose=False)

        # TODO: check to make sure the weights were properly quantized

        # save weights to h5:
        print("Saving quantized weights for next iteration...")
        yolo.save_weights(quant_weights_path_curr)

        # perform evaluation to see how badly pruning affected the accuracy
        print('Evaluating quantized network, before train step:')
        yolo.validate(train_imgs=train_imgs,
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


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
