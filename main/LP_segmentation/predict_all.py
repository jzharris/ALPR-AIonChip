#! /usr/bin/env python

import argparse
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes, crop_image
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

########################################################################################################################
# run: python predict_all.py -c config_lp_seg_mobilenet.json -w pruned_models\mobilenet_10it_20p_1\lp_seg_mobilenet_pruned_post-train_it9.h5 -i data\converted_dataset\test --crop
########################################################################################################################

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='folder of images or mp4 files')

argparser.add_argument(
    '--crop',
    action='store_true',
    default=False,
    help='crop the bounding box of each output')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_folder = args.input
    crop         = args.crop

    if crop:
        target_folder = "cropped"
    else:
        target_folder = "detected"

    detected_folder = os.path.join(image_folder, target_folder)
    if os.path.exists(detected_folder):
        shutil.rmtree(detected_folder)
    os.mkdir(detected_folder)

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    for root, dirs, files in os.walk(os.path.join(image_folder, "jpeg")):
        for file in tqdm(files):
            image_path = os.path.join(root, file)
            detected_path = os.path.join(detected_folder, file)

            if image_path[-4:] == '.mp4':
                video_out = image_path[:-4] + '_detected' + image_path[-4:]
                video_reader = cv2.VideoCapture(image_path)

                nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

                video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       50.0,
                                       (frame_w, frame_h))

                for i in tqdm(range(nb_frames)):
                    _, image = video_reader.read()

                    boxes = yolo.predict(image)
                    image = draw_boxes(image, boxes, config['model']['labels'])

                    video_writer.write(np.uint8(image))

                video_reader.release()
                video_writer.release()
            else:
                image = cv2.imread(image_path)
                boxes = yolo.predict(image)

                best_box = []
                for box in boxes:
                    if len(best_box) == 0:
                        best_box.append(box)
                    elif box.get_score() > best_box[0].get_score():
                        best_box[0] = box
                if not crop:
                    image = draw_boxes(image, best_box, config['model']['labels'])

                if len(best_box) == 0:
                    print(len(best_box), 'no boxes are chosen')
                elif crop:
                    cropped_image = crop_image(image, best_box[0])
                    cv2.imwrite(detected_path[:-4] + '' + detected_path[-4:], cropped_image)

                if not crop:
                    cv2.imwrite(detected_path[:-4] + '' + detected_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
