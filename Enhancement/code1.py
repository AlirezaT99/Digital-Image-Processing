import cv2 as cv
import numpy as np


def log_transform(value):
    value = 255 * np.log(1 + LOG_ALPHA * value) / np.log(1 + 255 * LOG_ALPHA)
    return value.astype(np.uint8)


def stretch_contrast(value):
    def transformation_function(x):
        if x <= R1:
            return (x ** 2) * S1 / (R1 ** 2)
        if x >= R2:
            return S2 + (255 - S2) * (x - R2) / (255 - x)
        return S1 + (S2 - S1) * (x - R1) / (R2 - R1)

    return np.vectorize(transformation_function)(value).astype(np.uint8)


FILE_PATH = 'Enhance1.JPG'
RES_PATH = 'results/res01.jpg'

LOG_ALPHA = 0.12
(R1, S1), (R2, S2) = (70, 70), (130, 225)
im = cv.imread(FILE_PATH, cv.IMREAD_COLOR)

im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
im[:, :, 2] = log_transform(im[:, :, 2])  # 2 is V Channel
im[:, :, 2] = stretch_contrast(im[:, :, 2])
im = cv.cvtColor(im, cv.COLOR_HSV2BGR)

cv.imwrite(RES_PATH, im)
