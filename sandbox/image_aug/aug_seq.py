import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import PIL.ImageOps

ia.seed(1)

seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # horizontal flips

    # iaa.Crop(percent=(0, 0.1)), # random crops

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
    ),

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

    # Add salt and pepper noise.
    iaa.SaltAndPepper(p=(0, 0.05)),

    # Add poisson noise.
    iaa.AdditivePoissonNoise(lam=(0, 10.0), per_channel=True),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
], random_order=False) # apply augmenters in random order

img = cv2.imread('res_inv.png')
images = []
for i in range(32):
    images.append(img)
images_aug = seq.augment_images(images)

for i in range(32):
    im = Image.fromarray(np.uint8(images_aug[i]))
    inverted_image = PIL.ImageOps.invert(im)
    fig, ax = plt.subplot(8, 4, i+1), plt.imshow(inverted_image)
    # ax.set_axis_off()
plt.show()
