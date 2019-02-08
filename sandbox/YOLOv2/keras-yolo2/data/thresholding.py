from skimage.filters import threshold_otsu, threshold_local
from scipy.misc import imread
import matplotlib.pyplot as plt

# Shows that global thresholding is the best option to use for our case


def threshold_img(image, threshold_type='global'):
    if threshold_type is 'global':
        global_thresh = threshold_otsu(image)
        binary = image > global_thresh
    elif threshold_type is 'local':
        block_size = 35
        local_thresh = threshold_local(image, block_size, offset=10)
        binary = image > local_thresh
    else:
        raise Exception('Wrong threshold_type')
    return binary


# image = imread('just_lps/─■A9H707.jpg', mode='L')
# image = imread('just_lps/├÷ASF227_27.jpg', mode='L')
# image = imread('just_lps/├╔HSB333_26.jpg', mode='L')
image = imread('just_lps/╛⌐LY3127_26.jpg', mode='L')

binary_global = threshold_img(image, 'global')
# global_thresh = threshold_otsu(image)
# binary_global = image > global_thresh

# block_size = 35
# local_thresh = threshold_local(image, block_size, offset=10)
# binary_local = image > local_thresh
binary_local = threshold_img(image, 'local')

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(image)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_local)
ax[2].set_title('Local thresholding')

for a in ax:
    a.axis('off')

plt.show()