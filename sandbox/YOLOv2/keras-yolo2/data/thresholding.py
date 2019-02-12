from skimage.filters import threshold_otsu, threshold_local
from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2

# Shows that global thresholding is the best option to use for our case


def threshold_img(image, threshold_type='global'):
    if threshold_type is 'global':
        img = cv2.bilateralFilter(image, 9, 75, 100)
        global_thresh = threshold_otsu(img)
        binary = img > global_thresh
    elif threshold_type is 'local':
        block_size = 101
        local_thresh = threshold_local(image, block_size, offset=10)
        binary = image > local_thresh
    elif threshold_type is 'adaptive':
        img = cv2.medianBlur(image, 1)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
    elif threshold_type is 'canny':
        img = cv2.bilateralFilter(image, 9, 75, 1000)
        th, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.Canny(img, th / 2, th)
    elif threshold_type is 'laplacian':
        img = cv2.GaussianBlur(image, (3, 3), 0)
        binary = cv2.Laplacian(img, cv2.CV_64F)
    elif threshold_type is 'bilateral':
        binary = cv2.bilateralFilter(image, 9, 75, 1000)
    elif threshold_type is 'none':
        binary = image
    else:
        raise Exception('Wrong threshold_type')
    return binary


def main():
    # image = imread('just_lps/─■A9H707.jpg', mode='L')
    # image = imread('just_lps/├÷ASF227_27.jpg', mode='L')
    # image = imread('just_lps/├╔HSB333_26.jpg', mode='L')
    # image = imread('just_lps/╛⌐LY3127_26.jpg', mode='L')
    image = imread('just_lps\─■A88888.jpg', mode='L')

    binary_global = threshold_img(image, 'global')
    binary_local = threshold_img(image, 'local')
    binary_adaptive = threshold_img(image, 'adaptive')

    fig, axes = plt.subplots(nrows=4, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()
    for a in ax:
        a.axis('off')

    ax[0].imshow(image)
    ax[0].set_title('Original')

    ax[1].imshow(binary_global)
    ax[1].set_title('Global thresholding')

    ax[2].imshow(binary_local)
    ax[2].set_title('Local thresholding')

    ax[3].imshow(binary_adaptive)
    ax[3].set_title('Adaptive thresholding')

    plt.show()


if __name__ == '__main__':
    main()
