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
import xml.etree.ElementTree

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

########################################################################################################################
# run: python predict.py -c config_lp_seg_mobilenet_quant.json
########################################################################################################################

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


# Convert XML to nested dictionary
def xml_to_dict(path):
    def xml_to_dict_recursive(node):
        if len(node)==0:
            return node.text
        return {child.tag:xml_to_dict_recursive(child) for child in node}
    return xml_to_dict_recursive(xml.etree.ElementTree.parse(path).getroot())


def _main_(args):
    config_path  = args.conf
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    image_folder = config['predict']['input']
    output_folder = config['predict']['output_folder']
    crop = config['predict']['crop']

    if crop:
        target_folder = "{}_cropped".format(output_folder)
    else:
        target_folder = "{}_detected".format(output_folder)

    detected_folder = os.path.join(image_folder, target_folder)
    if os.path.exists(detected_folder):
        shutil.rmtree(detected_folder)
    os.mkdir(detected_folder)

    weights_path = config['predict']['pretrained_weights']

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

    # dictionary of exported images used to generate sample.txt file:
    samples_dict = {}

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

                if crop:
                    best_box = []
                    for box in boxes:
                        if len(best_box) == 0:
                            best_box.append(box)
                        elif box.get_score() > best_box[0].get_score():
                            best_box[0] = box

                    if len(best_box) == 0:
                        print('no boxes are chosen')
                    else:
                        cropped_image = crop_image(image, best_box[0])

                        cv2.imwrite(detected_path[:-4] + '' + detected_path[-4:], cropped_image)

                        # find LP chars to add to sample.txt file
                        xml = xml_to_dict(os.path.join(image_folder, "xml", "{}.xml".format(file[:-4])))
                        samples_dict[file] = xml['object']['platetext']
                else:
                    image = draw_boxes(image, boxes, config['model']['labels'])
                    if len(boxes) == 0:
                        print('no boxes are chosen')

                    cv2.imwrite(detected_path[:-4] + '' + detected_path[-4:], image)

    if crop:
        # write sample.txt file:
        print('writing sample.txt file to {}'.format(os.path.join(detected_folder, 'sample.txt')))
        with open(os.path.join(detected_folder, 'sample.txt'), 'w+') as samples_txt:
            for key in samples_dict.keys():
                samples_txt.write("{} {}\n".format(key, samples_dict[key]))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
