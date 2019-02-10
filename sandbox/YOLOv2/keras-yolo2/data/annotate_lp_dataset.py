# run through entire lp dataset and count how many fail to be parsed by our technique

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from lp_bb import bb_img

###############################################################################################################
# Function for determining whether to accept the plate or not

possible_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'   # NO I's or O's exist in this dataset
char_counts = np.zeros(len(possible_chars))


def correct_letters(image, file, threshold_type='global', debug=False, output_dir=None):

    # plates letters to include in file names
    plate_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789_'

    # count how many unique candidates there are
    charCandidates, rects = bb_img(deepcopy(image), threshold_type)
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
        # Now that we have filtered out the outliers, perform pattern detection to
        # ensure the following pattern exists: X YYYYY
        percent_tol = 0.1  # X YYYYY - the Y's must have the same spacing
        centers = np.array([(x + w) / 2 for x, y, w, h in rects])
        arr1inds = centers.argsort()
        centers = centers[arr1inds[::-1]]
        # sort rects in ascending order:
        sorted_rects = []
        for idx in arr1inds:
            sorted_rects.append(rects[idx])
        rects = sorted_rects

        prev_dist = None
        for i in range(len(centers)):
            if i + 2 < len(centers):
                if prev_dist is None:
                    prev_dist = centers[i] - centers[i + 1]
                else:
                    curr_dist = centers[i] - centers[i + 1]
                    if np.abs(curr_dist - prev_dist) / prev_dist < percent_tol:
                        prev_dist = curr_dist
                    else:
                        # pattern failed, reject this one:
                        keep_plate = False
                        break

    if debug:
        print('{} ?= {}'.format(letter_count, correct_count))

    if keep_plate:
        if output_dir is not None:
            # check if this file already exists:
            output_file = path.join(output_dir, '{}.jpg'.format(digits))
            inc = 0
            while path.isfile(output_file):
                # add a new number to the end...for now
                output_file = path.join(output_dir, '{}_{}.jpg'.format(digits, inc))
                inc += 1

            cv2.imwrite(output_file, cv2.bitwise_or(image, charCandidates))

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

    return keep_plate, rects
###############################################################################################################


input_dir = 'just_lps'
out_dir = 'lp_candidates'

if not path.isdir(out_dir):
    os.mkdir(out_dir)

for root, dirs, files in os.walk(input_dir):
    for file in tqdm(files):
        file_path = path.join(input_dir, file)

        image = imread(file_path, mode='L')
        keep_plate, rects = correct_letters(image, file)
        if not keep_plate:
            # if incorrect, try again but invert the image
            image = cv2.bitwise_not(image)
            keep_plate, rects = correct_letters(image, file)
            if not keep_plate:
                # if still incorrect, try above pattern but with adaptive thresholding
                image = cv2.bitwise_not(image)
                keep_plate, rects = correct_letters(image, file, threshold_type='adaptive')
                if not keep_plate:
                    # last try: inverted image with adaptive thresholding
                    image = cv2.bitwise_not(image)
                    keep_plate, rects = correct_letters(image, file, threshold_type='adaptive')

        if keep_plate:
            # TODO: The rects array correspond to the plate characters. Create an xml file
            # TODO: to store each rect (use example.xml as a template)

            # TODO: rects will be in the following format:
            # TODO:     if the plate characters are "A 12345" then the first element in rects will be A and
            # TODO:     the last will be 5.

            # TODO: path.splitext(file)[0] will give the file name
            pass