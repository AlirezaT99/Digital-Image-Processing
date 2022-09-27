import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calc_scaled_cdf(x, bin_count=256):
    hist, _ = np.histogram(x, bins=bin_count)
    cdf = np.cumsum(hist)
    return cdf / cdf[-1] * 255


def save_hist(path, title, image):
    plt.title(title)
    plt.hist(image)
    plt.savefig(path, bbox_inches='tight')


def match_hist(image, target):
    target_cdf = calc_scaled_cdf(target)
    image_cdf = calc_scaled_cdf(image)
    cdf_x = np.vectorize(lambda x: image_cdf[x])
    cdf_1 = np.vectorize(lambda x: np.argmin(np.abs(target_cdf - x)))
    image = cdf_1(cdf_x(image))
    return image.astype(np.uint8)


FILE_PATH = 'Dark.jpg'
TEMPLATE_PATH = 'Pink.jpg'
OUT_HIST_PATH = 'results/res10.jpg'
OUT_PATH = 'results/res11.jpg'

im = cv.imread(FILE_PATH, cv.IMREAD_COLOR)
template = cv.imread(TEMPLATE_PATH, cv.IMREAD_COLOR)
im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
template = cv.cvtColor(template, cv.COLOR_BGR2HSV)

v = im[:, :, 2]
v = match_hist(v, template[:, :, 2])

save_hist(OUT_HIST_PATH, 'Result Hist', v)
im[:, :, 2] = v

im = cv.cvtColor(im, cv.COLOR_HSV2BGR)
cv.imwrite(OUT_PATH, im)
