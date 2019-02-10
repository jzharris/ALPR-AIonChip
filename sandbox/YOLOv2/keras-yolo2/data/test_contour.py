# run through entire lp dataset and count how many fail to be parsed by our technique
# achieves ~50% (1995 plates) with 1-letter deviation and 31% (1228 plates) with no letter deviation

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from lp_bb import bb_img, debug_bb

correct = 0
incorrect = 0


def correct_letters(image, threshold_type='global', debug=False, output_dir=None):

    plate_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'

    # count how many unique candidates there are
    charCandidates, rects = bb_img(deepcopy(image), threshold_type)
    letter_count = len(rects)

    # count how many are correctly contoured
    digits = ''
    passed_number = False
    for char in path.splitext(file)[0]:
        if char in plate_chars:
            digits = digits + char

    correct_count = 0
    for char in digits:
        if char is '_':
            break
        else:
            correct_count += 1

    # check if this file already exists:
    output_file = path.join(output_dir, '{}.jpg'.format(digits))
    inc = 0
    while path.isfile(output_file):
        # add a new number to the end...for now
        output_file = path.join(output_dir, '{}_{}.jpg'.format(digits, inc))
        inc += 1

    if debug:
        print('{} ?= {}'.format(letter_count, correct_count))
    return_val = correct_count - 1 <= letter_count <= correct_count + 1

    if return_val:
        if output_dir is not None:
            cv2.imwrite(output_file, cv2.bitwise_or(image, charCandidates))

    return return_val


debug = False
export = True
show_correct = True
show_incorrect = False
show_steps = True
lp_dir = 'just_lps'
out_dir = 'lp_candidates'

if not path.isdir(out_dir):
    os.mkdir(out_dir)

for root, dirs, files in os.walk(lp_dir):
    for file in tqdm(files):
        file_path = path.join(lp_dir, file)
        if debug:
            print(file_path)

        image = imread(file_path, mode='L')
        if correct_letters(image, debug=debug, output_dir=(out_dir if export else None)):
            if debug and show_correct:
                debug_bb(image, show_steps=show_steps)
            correct += 1
        else:
            # if incorrect, try again but invert the image
            image = cv2.bitwise_not(image)
            if correct_letters(image, debug=debug, output_dir=(out_dir if export else None)):
                if debug and show_correct:
                    debug_bb(image, show_steps=show_steps)
                correct += 1
            else:
                # if still incorrect, try above pattern but with adaptive thresholding
                image = cv2.bitwise_not(image)
                if correct_letters(image, threshold_type='adaptive', debug=debug,
                                   output_dir=(out_dir if export else None)):
                    if debug and show_correct:
                        debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
                    correct += 1
                else:
                    image = cv2.bitwise_not(image)
                    if correct_letters(image, threshold_type='adaptive', debug=debug,
                                       output_dir=(out_dir if export else None)):
                        if debug and show_correct:
                            debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
                        correct += 1
                    else:
                        if debug and show_incorrect:
                            debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
                        incorrect += 1


# print accuracy
print("Accuracy: {} ({}/{})".format(correct / (correct + incorrect), correct, correct + incorrect))

# # plot histogram of counts
# plt.hist(np.array(counts), bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.title("Histogram of letter counts")
# plt.show()
