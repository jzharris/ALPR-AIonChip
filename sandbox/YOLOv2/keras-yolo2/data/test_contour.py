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


correct = 0
incorrect = 0

export = False
lp_dir = 'converted_dataset2/train/jpeg'
out_dir = 'lp_candidates_d1'
if not path.isdir(out_dir):
    os.mkdir(out_dir)

for root, dirs, files in os.walk(lp_dir):
    for file in tqdm(files):
        file_path = path.join(lp_dir, file)

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

        keep_plate = correct_letters(file, rects)
        if keep_plate:
            correct += 1
        else:
            incorrect += 1

# print accuracy
print("Accuracy: {} ({}/{})".format(correct / (correct + incorrect), correct, correct + incorrect))

# print(char_counts)
hist_labels = []
for char in possible_chars:
    hist_labels.append(char)
plt.bar(np.arange(len(char_counts)), char_counts, tick_label=hist_labels)
plt.title('Number of (extracted) character occurences in LP dataset')
plt.show()
