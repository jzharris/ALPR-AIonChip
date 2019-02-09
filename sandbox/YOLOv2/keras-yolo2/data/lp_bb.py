from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from thresholding import threshold_img


def bb_img(image, threshold_type='global'):

    # apply thresholding
    thresh = threshold_img(image, threshold_type).astype(np.uint8)

    # apply contouring
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.zeros(len(contours))
    for i, cnt in enumerate(contours):
        areas[i] = cv2.contourArea(cnt)
    avg_area = np.average(areas)

    # remove extremely large contours
    max_tolerance = 1
    min_tolerance = 1
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if avg_area - (avg_area - np.min(areas)) * min_tolerance <= areas[i] <= (
                np.max(areas) - avg_area) * max_tolerance + avg_area:
            filtered_contours.append(cnt)

    ##################################################################################
    # Row 4:
    charCandidates = np.zeros(thresh.shape, dtype="uint8")

    c = max(filtered_contours, key=cv2.contourArea)
    for i, cnt in enumerate(filtered_contours):
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(image.shape[0])

        keepAspectRatio = 0.3 < aspectRatio < 0.8
        keepSolidity = solidity > 0.15
        keepHeight = 0.4 < heightRatio < 0.95

        if keepAspectRatio and keepSolidity and keepHeight:
            # hull = cv2.convexHull(cnt)
            # cv2.drawContours(charCandidates, [hull], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(charCandidates, (x, y), (x + w, y + h), 255, 1)

    return charCandidates


def debug_bb(image, threshold_type='global'):
    fig, axes = plt.subplots(nrows=4, figsize=(7, 8))
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
    # apply thresholding
    thresh = threshold_img(image, threshold_type).astype(np.uint8)
    ax[1].imshow(thresh)
    ax[1].set_title('{} thresholding'.format(threshold_type))

    ##################################################################################
    # Row 3:
    # apply contouring
    cont_image = copy.deepcopy(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.zeros(len(contours))
    for i, cnt in enumerate(contours):
        areas[i] = cv2.contourArea(cnt)
    avg_area = np.average(areas)

    # remove extremely large contours
    max_tolerance = 1
    min_tolerance = 1
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if avg_area - (avg_area - np.min(areas)) * min_tolerance <= areas[i] <= (
                np.max(areas) - avg_area) * max_tolerance + avg_area:
            cv2.drawContours(cont_image, [cnt], 0, (255, 255, 255), 3)
            filtered_contours.append(cnt)

    ax[2].imshow(cont_image)
    ax[2].set_title('Contoured')

    ##################################################################################
    # Row 4:
    charCandidates = np.zeros(thresh.shape, dtype="uint8")

    c = max(filtered_contours, key=cv2.contourArea)
    for i, cnt in enumerate(filtered_contours):
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(image.shape[0])

        keepAspectRatio = 0.3 < aspectRatio < 0.8
        keepSolidity = solidity > 0.15
        keepHeight = 0.4 < heightRatio < 0.95

        if keepAspectRatio and keepSolidity and keepHeight:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(charCandidates, [hull], -1, 200, -1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(charCandidates, (x, y), (x + w, y + h), 255, 1)

    ax[3].imshow(charCandidates)
    ax[3].set_title('Candidates')

    plt.show()


def main():
    image = imread('just_lps/├÷D88888_10.jpg', mode='L')
    debug_bb(image)


if __name__ == '__main__':
    main()
