import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from imgaug import augmenters as iaa

img = cv2.imread('res_orig.png')
rows, cols, ch = img.shape

####################################################################################
# Add skew to image

shear = 30
pad = 50
pts1 = np.float32([[pad, 0], [pad+10, 0], [cols-pad, rows]])
pts2 = np.float32([[pad - shear, 0], [pad + 10 - shear, 0], [cols - pad + shear, rows]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

####################################################################################
# Add noise

# gaussian noise
aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)
noise1 = aug.augment_image(dst)

# poisson noise
aug = iaa.AdditivePoissonNoise(lam=10.0, per_channel=True)
noise2 = aug.augment_image(dst)

# salt and pepper noise
aug = iaa.SaltAndPepper(p=0.05)
noise3 = aug.augment_image(dst)

# im = Image.fromarray(im_arr).convert('RGB')
# im.show()

####################################################################################
# Display

plt.subplot(231), plt.imshow(img), plt.title('Input')
plt.subplot(232), plt.imshow(dst), plt.title('Output')
plt.subplot(233), plt.imshow(noise1), plt.title('Gaussian')
plt.subplot(234), plt.imshow(noise2), plt.title('Poisson')
plt.subplot(235), plt.imshow(noise3), plt.title('S&P')
plt.show()
