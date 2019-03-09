#! /usr/bin/env python

import argparse
import os
import sys
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json

from pruning.prune_network import prune_layers, check_pruned_weights, print_pruned_weights
import keras.backend as K

##########################################################################################################
# run: python train.py -c config_lp_seg_mobilenet.json 2>&1 | tee pruned_models\mobilenet_8it_20p\logs.txt
##########################################################################################################

iterations = 8
epochs = [None, 1, 1, 2, 2, 2, 3, 4, 5]
skip_first_train = True

prune_threshold = 0.2
white_list = [] #['DetectionLayer/kernel:0']
white_regex = ['bias', 'gamma', 'beta', 'CustomAdam', 'loss', 'running_mean', 'running_variance',
               'moving_mean', 'moving_variance', 'DetectionLayer']

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
        train_valid_split = int(0.8*len(train_imgs))
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

    for it in range(iterations):
        print("it {}".format(it))

        # update grad_mask_consts if it > 0
        if it > 0 and grad_mask_consts is not None:
            yolo.grad_mask_consts = grad_mask_consts
            yolo.recompile(train_imgs         = train_imgs,
                           valid_imgs         = valid_imgs,
                           train_times        = config['train']['train_times'],
                           valid_times        = config['valid']['valid_times'],
                           learning_rate      = config['train']['learning_rate'],
                           batch_size         = config['train']['batch_size'],
                           warmup_epochs      = config['train']['warmup_epochs'],
                           object_scale       = config['train']['object_scale'],
                           no_object_scale    = config['train']['no_object_scale'],
                           coord_scale        = config['train']['coord_scale'],
                           class_scale        = config['train']['class_scale'],
                           debug              = config['train']['debug'])

        ###############################
        #   Load the pretrained weights (if any)
        ###############################

        if it > 0 and os.path.exists(config['train']['pruned_weights_name']+"_it{}.h5".format(it-1)):
            print("Loading pruned weights in", config['train']['pruned_weights_name']+"_it{}.h5".format(it-1))
            yolo.load_weights(config['train']['pruned_weights_name']+"_it{}.h5".format(it-1))

            # check to make sure the weights are pruned
            sess = K.get_session()
            check_pruned_weights(sess, grad_mask_consts, prune_threshold, it-1)
        elif os.path.exists(config['train']['pretrained_weights']):
            print("Loading pre-trained weights in", config['train']['pretrained_weights'])
            yolo.load_weights(config['train']['pretrained_weights'])
        else:
            print("No pre-trained weights were loaded")

        ###############################
        #   Start the training process
        ###############################

        if it != 0 or not skip_first_train:

            yolo.train(train_imgs         = train_imgs,
                       valid_imgs         = valid_imgs,
                       train_times        = config['train']['train_times'],
                       valid_times        = config['valid']['valid_times'],
                       nb_epochs          = epochs[it],
                       learning_rate      = config['train']['learning_rate'],
                       batch_size         = config['train']['batch_size'],
                       warmup_epochs      = config['train']['warmup_epochs'],
                       object_scale       = config['train']['object_scale'],
                       no_object_scale    = config['train']['no_object_scale'],
                       coord_scale        = config['train']['coord_scale'],
                       class_scale        = config['train']['class_scale'],
                       saved_weights_name = config['train']['saved_weights_name']+"_it{}.h5".format(it),
                       debug              = config['train']['debug'])
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

        ################################################################################################################
        # Calculate grad_mask_consts

        sess = K.get_session()
        if it > 0:
            # show the weights just after training
            check_pruned_weights(sess, grad_mask_consts, prune_threshold, it-1)
            sys.stdout.flush()
        grad_mask_consts = prune_layers(sess, prune_threshold, grad_mask_consts, white_list, white_regex,
                                        verbose=config['train']['verbose'])
        sys.stdout.flush()
        if config['train']['verbose']:
            print_pruned_weights(sess, grad_mask_consts)
            sys.stdout.flush()
        check_pruned_weights(sess, grad_mask_consts, prune_threshold, it)
        sys.stdout.flush()
        print('='*20)

        # save weights to h5:
        if config['train']['pruned_weights_name']:
            print("Saving pruned weights for next iteration...")
            yolo.save_weights(config['train']['pruned_weights_name']+"_it{}.h5".format(it))

        # perform evaluation to see how badly pruning affected the accuracy
        print('Evaluating pruned network, before train step:')
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
        sys.stdout.flush()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
