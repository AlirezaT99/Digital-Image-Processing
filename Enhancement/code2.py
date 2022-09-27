import cv2 as cv
import numpy as np


def log_transform(image):
    image = 255 * np.log(1 + LOG_ALPHA * image) / np.log(1 + 255 * LOG_ALPHA)
    return image.astype(np.uint8)


FILE_PATH = 'Enhance2.JPG'
RES_PATH = 'results/res02.jpg'

LOG_ALPHA = 0.1

im = cv.imread(FILE_PATH, cv.IMREAD_COLOR)

im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
im[:, :, 2] = log_transform(im[:, :, 2])
im = cv.cvtColor(im, cv.COLOR_HSV2BGR)

cv.imwrite(RES_PATH, im)
