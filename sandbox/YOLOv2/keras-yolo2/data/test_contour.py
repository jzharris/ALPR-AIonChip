# run through entire lp dataset and count how many fail to be parsed by our technique
# achieves ~50% (1995 plates) with 1-letter deviation and 31% (1228 plates) with no letter deviation

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lp_bb import bb_img, debug_bb

correct = 0
incorrect = 0


def correct_letters(image, threshold_type='global'):

    plate_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # get candidates from license plate
    candidates = bb_img(image, threshold_type)

    # count how many unique candidates there are
    contours, hierarchy = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_count = len(contours)
    # counts.append(letter_count)

    # count how many are correctly contoured
    correct_count = 0
    for char in path.splitext(file)[0]:
        if char == '_':
            break
        if char in plate_chars:
            correct_count += 1

    return correct_count <= letter_count <= correct_count


debug = True
show_correct = False
lp_dir = 'just_lps'

for root, dirs, files in os.walk(lp_dir):
    for file in tqdm(files):
        file_path = path.join(lp_dir, file)

        image = imread(file_path, mode='L')
        if correct_letters(image):
            if debug and show_correct:
                debug_bb(image)
            correct += 1
        else:
            # if incorrect, try again but invert the image
            image = cv2.bitwise_not(image)
            if correct_letters(image):
                if debug and show_correct:
                    debug_bb(image)
                correct += 1
            else:
                # if still incorrect, try adaptive thresholding
                image = cv2.bitwise_not(image)
                if correct_letters(image, threshold_type='adaptive'):
                    if debug and show_correct:
                        debug_bb(image, threshold_type='adaptive')
                    correct += 1
                else:
                    image = cv2.bitwise_not(image)
                    if correct_letters(image, threshold_type='adaptive'):
                        if debug and show_correct:
                            debug_bb(image, threshold_type='adaptive')
                        correct += 1
                    else:
                        if debug and not show_correct:
                            debug_bb(image)
                        incorrect += 1


# print accuracy
print("Accuracy: {} ({}/{})".format(correct / (correct + incorrect), correct, correct + incorrect))

# # plot histogram of counts
# plt.hist(np.array(counts), bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.title("Histogram of letter counts")
# plt.show()
