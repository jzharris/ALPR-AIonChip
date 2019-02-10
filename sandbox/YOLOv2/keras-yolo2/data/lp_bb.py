from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from thresholding import threshold_img


def get_aspect(cnt):
    (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
    return boxW / float(boxH)


def get_heightr(cnt, image):
    (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
    return boxH / float(image.shape[0])


#TODO: if all bb's start/end at a specific height (except 1), then crop the outlier to that height - use std dev to tell the difference
def bb_img(image, threshold_type='global'):

    # apply thresholding
    thresh = threshold_img(image, threshold_type).astype(np.uint8)

    # apply contouring
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.zeros(len(contours))
    aspects = np.zeros(len(contours))
    heights = np.zeros(len(contours))
    for i, cnt in enumerate(contours):
        areas[i] = cv2.contourArea(cnt)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
        aspectRatio = boxW / float(boxH)
        aspects[i] = aspectRatio
        heights[i] = boxH / float(image.shape[0])

    # remove area outliers
    mean = np.average(areas)
    std = np.std(areas)
    N = 3
    filtered_contours = [x for x in contours if (mean + N * std > cv2.contourArea(x) > mean - N * std)]
    # filtered_contours = contours

    # # remove aspect outliers
    mean = np.average(aspects)
    std = np.std(aspects)
    N = 3
    filtered_contours = [x for x in filtered_contours if (mean + N * std > get_aspect(x) > mean - N * std)]

    # remove height outliers
    mean = np.average(heights)
    std = np.std(heights)
    # if std > 0.3:
    N = 0.7
    filtered_contours = [x for x in filtered_contours if (mean + N * std > get_heightr(x, image) > mean - N * std)]

    ##################################################################################
    # Row 4:
    charCandidates = np.zeros(thresh.shape, dtype="uint8")

    rects = []
    if len(filtered_contours) > 0:
        c = max(filtered_contours, key=cv2.contourArea)
        for i, cnt in enumerate(filtered_contours):
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(image.shape[0])
            right_bound = boxX + boxW
            rightRatio = right_bound / float(image.shape[1])

            keepAspectRatio = 0.2 < aspectRatio < 0.8 # might affect some I's
            keepSolidity = solidity > 0.15
            keepHeight = 0.4 < heightRatio < 0.9
            keepRight = rightRatio > 0.2    # try to avoid left-most characters
            keepLocalized = right_bound <= image.shape[1] - 1

            if keepAspectRatio and keepSolidity and keepHeight and keepRight and keepLocalized:
                # hull = cv2.convexHull(cnt)
                # cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append([x, y, w, h])
                cv2.rectangle(charCandidates, (x, y), (x + w, y + h), 255, 1)

    return charCandidates, rects


def debug_bb(image, threshold_type='global', show_steps=False):
    if show_steps:
        fig, axes = plt.subplots(nrows=4, figsize=(7, 8))
        ax = axes.ravel()
        plt.gray()
        for a in ax:
            a.axis('off')

        ##################################################################################
        # Row 1:
        ax[0].imshow(image)
        ax[0].set_title('Original')

        ##################################################################################
        # Row 2:
        # apply thresholding
        thresh = threshold_img(image, threshold_type).astype(np.uint8)
        ax[1].imshow(thresh)
        ax[1].set_title('{} thresholding'.format(threshold_type))

        ##################################################################################
        # Row 3:
        # apply contouring
        cont_image = copy.deepcopy(image)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            cv2.drawContours(cont_image, [cnt], 0, (255, 255, 255), 3)

        ax[2].imshow(cont_image)
        ax[2].set_title('Contoured')

        ##################################################################################
        # Row 4:
        charCandidates = bb_img(image, threshold_type)
        ax[3].imshow(cv2.bitwise_or(image, charCandidates))
        ax[3].set_title('Candidates')

        plt.show()
    else:
        fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
        ax = axes.ravel()
        plt.gray()
        for a in ax:
            a.axis('off')

        ##################################################################################
        # Row 1:
        # image = imread('just_lps/├╔HSB333_26.jpg', mode='L')
        # image = imread('just_lps/─■A9H707.jpg', mode='L')
        # image = imread(filename, mode='L')
        ax[0].imshow(image)
        ax[0].set_title('Original')

        ##################################################################################
        # Row 2:
        charCandidates = bb_img(image, threshold_type)
        ax[1].imshow(cv2.bitwise_or(image, charCandidates))
        ax[1].set_title('Candidates')

        plt.show()


def main():
    image = imread('just_lps/├÷D88888_10.jpg', mode='L')
    debug_bb(image)


if __name__ == '__main__':
    main()
