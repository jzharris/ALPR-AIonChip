# run through entire lp dataset and count how many fail to be parsed by our technique

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from skimage.filters import threshold_local

possible_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'   # NO I's or O's exist in this dataset
char_counts = np.zeros(len(possible_chars))


def draw_contours(image, contours):
    if len(contours) > 0:
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 1)

    return image


def filter_image(image):
    block_size = 101
    local_thresh = threshold_local(image, block_size, offset=10)
    thresh = (image <= local_thresh).astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image = draw_contours(image, contours)
    return image, contours


def filter_contours(image, contours):
    rects = []
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        for i, cnt in enumerate(contours):
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(image.shape[0])

            keepAspectRatio = 0.1 < aspectRatio < 0.95
            keepSolidity = solidity > 0.15
            keepHeight = 0.3 < heightRatio < 0.9

            if keepAspectRatio and keepSolidity and keepHeight:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, w, h))

    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, 1)

    return image, rects


def sort_rects(rects):
    centers = np.array([(x + w) / 2 for x, y, w, h in rects])
    arr1inds = centers.argsort()

    sorted_rects = []
    for idx in arr1inds:
        sorted_rects.append(rects[idx])

    return sorted_rects


def correct_letters(file, rects):

    # plates letters to include in file names
    plate_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789_'

    # count how many unique candidates there are
    letter_count = len(rects)

    # count how many are correctly contoured
    digits = ''
    for char in path.splitext(file)[0]:
        if char in plate_chars:
            digits = digits + char

    correct_count = 0
    for char in digits:
        if char is '_':
            break
        else:
            correct_count += 1

    keep_plate = correct_count <= letter_count <= correct_count
    if keep_plate:
        global char_counts
        global possible_chars
        if possible_chars is not None:
            for char in digits:
                if char is '_':
                    break
                else:
                    found = possible_chars.find(char)
                    if found is not -1:
                        char_counts[found] += 1

    return keep_plate


def annotate(type='train', export=True):
    correct = 0
    incorrect = 0

    input_dir = 'converted_dataset2/{}/jpeg'.format(type)
    out_dir = 'lp_candidates'
    jpg_dir = path.join(out_dir, type, 'jpeg')
    xml_dir = path.join(out_dir, type, 'xml')

    if not path.isdir(jpg_dir):
        os.makedirs(jpg_dir)
    if not path.isdir(xml_dir):
        os.makedirs(xml_dir)

    file_counter = 0

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            file_path = path.join(input_dir, file)

            # Proven to be devoid of I's and O's
            # if "I" in file:
            #     print("I")
            #     exit(0)
            #
            # if "O" in file:
            #     print("O")
            #     exit(0)

            image = imread(file_path, mode='L')

            img = deepcopy(image)
            output_img, contours = filter_image(img)
            # plt.imshow(output_img)
            # plt.show()

            img = deepcopy(image)
            filtered_img, rects = filter_contours(img, contours)
            # plt.imshow(filtered_img)
            # plt.show()

            rects = sort_rects(rects)

            keep_plate = correct_letters(file, rects)

            if keep_plate:
                correct += 1

                if export:
                    # get chars from file name
                    plate_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789_'
                    chars = ''
                    for char in path.splitext(file)[0]:
                        if char in plate_chars:
                            chars = chars + char

                    # create set of bb's for xml file
                    bb_set = ''
                    with open('bb_xml.txt', 'r') as bb_file:
                        bb_xml_template = bb_file.read()
                        for idx, (x, y, w, h) in enumerate(rects):
                            xmin = x
                            ymin = y
                            xmax = x + w
                            ymax = y + h

                            bb_xml_item = bb_xml_template.format(chars[idx], xmin, ymin, xmax, ymax)
                            bb_set = bb_set + bb_xml_item + '\n'

                    # open template file and use as base for new file:
                    with open('template_xml.txt', 'r') as myfile:
                        ###########################################################################################
                        # Do for original color image:
                        data = myfile.read()
                        formatted = data.format('{}.jpg'.format(file_counter), image.shape[1], image.shape[0], bb_set)

                        # save formatted to a new xml file
                        with open(path.join(xml_dir, '{}.xml'.format(file_counter)), 'w+') as xml_file:
                            xml_file.write(formatted)

                        # save the (colored) corresponding image as new jpg file
                        cv2.imwrite(path.join(jpg_dir, '{}.jpg'.format(file_counter)), imread(file_path))

                        ###########################################################################################
                        # Do for B/W color image:
                        formatted = data.format('{}_bw.jpg'.format(file_counter), image.shape[1], image.shape[0],
                                                bb_set)

                        # save formatted to a new xml file
                        with open(path.join(xml_dir, '{}_bw.xml'.format(file_counter)), 'w+') as xml_file:
                            xml_file.write(formatted)

                        # save the (colored) corresponding image as new jpg file
                        cv2.imwrite(path.join(jpg_dir, '{}_bw.jpg'.format(file_counter)), imread(file_path, mode='L'))

                        ###########################################################################################
                        # Do for inverted B/W color image:
                        formatted = data.format('{}_inv.jpg'.format(file_counter), image.shape[1], image.shape[0],
                                                bb_set)

                        # save formatted to a new xml file
                        with open(path.join(xml_dir, '{}_inv.xml'.format(file_counter)), 'w+') as xml_file:
                            xml_file.write(formatted)

                        # save the (colored) corresponding image as new jpg file
                        cv2.imwrite(path.join(jpg_dir, '{}_inv.jpg'.format(file_counter)),
                                    255 - imread(file_path, mode='L'))

                    file_counter += 1

            else:
                incorrect += 1

    # print accuracy
    print("Accuracy: {} ({}/{})".format(correct / (correct + incorrect), correct, correct + incorrect))

    if not export:
        # print(char_counts)
        hist_labels = []
        for char in possible_chars:
            hist_labels.append(char)
        plt.bar(np.arange(len(char_counts)), char_counts, tick_label=hist_labels)
        plt.title('Number of (extracted) character occurences in LP dataset')
        plt.show()


if __name__ == '__main__':
    annotate('train', True)
    annotate('test', True)
