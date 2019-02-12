# run through entire lp dataset and count how many fail to be parsed by our technique

import os
import os.path as path
from scipy.misc import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from skimage.filters import threshold_otsu, threshold_local

from lp_bb import bb_img, debug_bb

possible_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'   # NO I's or O's exist in this dataset
char_counts = np.zeros(len(possible_chars))


def apply_filter(image):
    block_size = 101
    local_thresh = threshold_local(image, block_size, offset=10)
    thresh = (image <= local_thresh).astype(np.uint8)
    # thresh = debug_bb(image, threshold_type='local', show_steps=show_steps)
    # thresh = cv2.bitwise_not(binary)
    # debug_bb(thresh, threshold_type='none', show_steps=show_steps)
    charCandidates = np.zeros(thresh.shape, dtype="uint8")
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for i, cnt in enumerate(contours):
    #     cv2.drawContours(charCandidates, [cnt], 0, (255, 255, 255), 3)

    rects = []
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
    for x, y, w, h in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, 1)

    return image, rects


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


correct = 0
incorrect = 0
export = False

debug = True
show_steps = True
show_correct = False
show_incorrect = True

lp_dir = 'converted_dataset2/train/jpeg'
out_dir = 'lp_candidates_d1'

if not path.isdir(out_dir):
    os.mkdir(out_dir)

for root, dirs, files in os.walk(lp_dir):
    for file in tqdm(files):
        file_path = path.join(lp_dir, file)
        if debug:
            print(file_path)

        # Proven to be devoid of I's and O's
        # if "I" in file:
        #     print("I")
        #     exit(0)
        #
        # if "O" in file:
        #     print("O")
        #     exit(0)

        image = imread(file_path, mode='L')
        output_img, rects = apply_filter(image)

        plt.imshow(output_img)
        plt.show()
        exit(0)




        # keep_plate, rects = correct_letters(image, file, debug=debug, output_dir=(out_dir if export else None))
        # if not keep_plate:
        #     debug_bb(image, threshold_type='canny', show_steps=show_steps)
        #     exit(0)
        #     # if incorrect, try again but invert the image
        #     image = cv2.bitwise_not(image)
        #     keep_plate, rects = correct_letters(image, file, debug=debug, output_dir=(out_dir if export else None))
        #     if not keep_plate:
        #         # if still incorrect, try above pattern but with adaptive thresholding
        #         image = cv2.bitwise_not(image)
        #         keep_plate, rects = correct_letters(image, file, threshold_type='adaptive',
        #                                             debug=debug, output_dir=(out_dir if export else None))
        #         if not keep_plate:
        #             # last try: inverted image with adaptive thresholding
        #             image = cv2.bitwise_not(image)
        #             keep_plate, rects = correct_letters(image, file, threshold_type='adaptive',
        #                                                 debug=debug, output_dir=(out_dir if export else None))
        #             if not keep_plate:
        #                 if show_incorrect:
        #                     debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
        #             else:
        #                 if show_correct:
        #                     debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
        #         elif show_correct:
        #             debug_bb(image, threshold_type='adaptive', show_steps=show_steps)
        #     elif show_correct:
        #         debug_bb(image, show_steps=show_steps)
        # elif show_correct:
        #     debug_bb(image, show_steps=show_steps)
        #
        # if keep_plate:
        #     correct += 1
        # else:
        #     incorrect += 1

# print accuracy
print("Accuracy: {} ({}/{})".format(correct / (correct + incorrect), correct, correct + incorrect))

# print(char_counts)
hist_labels = []
for char in possible_chars:
    hist_labels.append(char)
plt.bar(np.arange(len(char_counts)), char_counts, tick_label=hist_labels)
plt.title('Number of (extracted) character occurences in LP dataset')
plt.show()
