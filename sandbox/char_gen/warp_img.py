import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('res.png')
rows, cols, ch = img.shape

skew = 30
pad = 50
pts1 = np.float32([[pad, 0], [pad+10, 0], [cols-pad, rows]])
pts2 = np.float32([[pad-skew, 0], [pad+10-skew, 0], [cols-pad + skew, rows]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
