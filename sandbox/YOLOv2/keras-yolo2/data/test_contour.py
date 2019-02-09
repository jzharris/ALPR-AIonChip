# run through entire lp dataset and count how many fail to be parsed by our technique

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt

from lp_bb import bb_img, debug_bb

# counts = []
correct = 0
incorrect = 0
plate_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def correct_letters(image):
    # get candidates from license plate
    candidates = bb_img(image)

    # count how many unique candidates there are
    contours, hierarchy = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_count = len(contours)
    # counts.append(letter_count)

    # count how many are correctly contoured
    correct_count = 0
    for char in path.splitext(file)[0]:
        if char in plate_chars:
            correct_count += 1

    return correct_count <= letter_count <= correct_count + 2


lp_dir = 'just_lps'
for root, dirs, files in os.walk(lp_dir):
    for file in files:
        file_path = path.join(lp_dir, file)

        image = imread(file_path, mode='L')
        if correct_letters(image):
            correct += 1
        else:
            # if incorrect, try again but invert the image
            print(np.min(image), np.max(image))
            image = cv2.bitwise_not(image)
            if correct_letters(image):
                correct += 1
            else:
                debug_bb(cv2.bitwise_not(image))
                debug_bb(image)
                incorrect += 1
                exit(0)


# print accuracy
print("Accuracy: {}".format(correct / (correct + incorrect)))

# # plot histogram of counts
# plt.hist(np.array(counts), bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.title("Histogram of letter counts")
# plt.show()
