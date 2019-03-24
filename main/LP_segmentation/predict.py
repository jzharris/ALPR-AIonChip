#! /usr/bin/env python

import argparse
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes, crop_image, bbox_iou2
from frontend import YOLO
import json
from xml.etree import ElementTree

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
    return xml_to_dict_recursive(ElementTree.parse(path).getroot())


def xml_to_dict2(path):
    def xml_to_dict_recursive(node):
        if len(node)==0:
            return node.text
        return {child.tag:xml_to_dict_recursive(child) for child in node}
    tree = ElementTree.parse(path)
    parsed_objs = []
    for node in tree.iter('annotation'):
        objects = node.findall('object')
        for i, object in enumerate(objects):
            parsed = xml_to_dict_recursive(object)
            parsed_objs.append(parsed)

    xml_dict = xml_to_dict(path)
    xml_dict['object'] = parsed_objs
    return xml_dict


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
    found_boxes = 0
    total_boxes = 0

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
                    xml_path = os.path.join(image_folder, "xml", "{}.xml".format(file[:-4]))

                    # give benefit of the doubt to the bounding boxes, and add them if they overlap very well with the other bb's

                    # find LP chars to add to sample.txt file
                    xml = xml_to_dict2(xml_path)
                    objects = xml['object']
                    # if len(objects) > 1:
                    #     for object in objects:
                    #         print(object)
                    #     exit(0)

                    # iterate over every bounding box:
                    z = 0
                    total_boxes += len(objects)
                    for truth_object in objects:
                        truth_box = truth_object['bndbox']
                        box_found = False
                        for box in boxes:
                            iou = bbox_iou2(image, truth_box, box)

                            if iou > 0.0:
                                # the boxes overlap, this must be truth. Crop this part of the image
                                filename = os.path.splitext(file)[0]
                                ext = os.path.splitext(file)[1]
                                output_key = '{}_{}{}'.format(filename, z, ext)
                                samples_dict[output_key] = truth_object['platetext']

                                output_file = detected_path[:-4] + '_{}'.format(z) + detected_path[-4:]
                                cropped_image = crop_image(image, box)
                                cv2.imwrite(output_file, cropped_image)
                                found_boxes += 1
                                z += 1
                                break
                else:
                    image = draw_boxes(image, boxes, config['model']['labels'])
                    if len(boxes) == 0:
                        print('no boxes are chosen')

                    cv2.imwrite(detected_path[:-4] + '' + detected_path[-4:], image)

    if crop:
        # write sample.txt file:
        print('writing sample.txt file to {}'.format(os.path.join(detected_folder, 'sample.txt')))
        with open(os.path.join(detected_folder, 'sample.txt'), 'w+') as samples_txt:
            for idx, key in enumerate(samples_dict.keys()):
                if samples_dict[key]:
                    if idx > 0:
                        samples_txt.write("\n")
                    samples_txt.write("{} {}".format(key, samples_dict[key]))

        print('accuracy of cropping: {}/{} = {}%'.format(found_boxes, total_boxes, found_boxes / total_boxes * 100))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
