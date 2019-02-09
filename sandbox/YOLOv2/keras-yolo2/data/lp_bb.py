from scipy.misc import imread, toimage
import matplotlib.pyplot as plt
import cv2
import numpy as np

from thresholding import threshold_img

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

# image = imread('just_lps/├╔HSB333_26.jpg', mode='L')
image = imread('just_lps/─■A9H707.jpg', mode='L')
ax[0].imshow(image)
ax[0].set_title('Original')

# apply thresholding
thresh = threshold_img(image, 'global').astype(np.uint8)
ax[1].imshow(thresh)
ax[1].set_title('Global thresholding')

# apply contouring
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = np.zeros(len(contours))
for i, cnt in enumerate(contours):
    areas[i] = cv2.contourArea(cnt)

avg_area = np.average(areas)
for i, cnt in enumerate(contours):
    if areas[i] > avg_area:
        cv2.drawContours(image, [cnt], 0, (255, 255, 255), 3)

ax[2].imshow(image)
ax[2].set_title('Contoured')

for a in ax:
    a.axis('off')

plt.show()
