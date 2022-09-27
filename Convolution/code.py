import time

import cv2 as cv
import numpy as np
from scipy.signal import convolve2d

FILE_PATH = 'Pink.jpg'
RES_PATHS = ['results/res07.jpg', 'results/res08.jpg', 'results/res09.jpg']

BOX_FILTER_SIZE = 3


def method_1(image):
    frame = BOX_FILTER_SIZE // 2
    box = np.ones((BOX_FILTER_SIZE, BOX_FILTER_SIZE)) / (BOX_FILTER_SIZE ** 2)
    result = np.zeros((image.shape[0] - 2 * frame, image.shape[1] - 2 * frame, 3))
    for ch in range(3):
        result[:, :, ch] = convolve2d(image[:, :, ch], box, mode='valid')
    return result.astype(np.uint8)


def method_2(image):
    radius = BOX_FILTER_SIZE // 2
    result = np.zeros(image.shape)

    def get_box_mean(x, y):
        x_min, x_max = x - radius, x + radius + 1
        y_min, y_max = y - radius, y + radius + 1
        sub_image = image[x_min:x_max, y_min:y_max]
        return np.array([np.mean(sub_image[:, :, 0]), np.mean(sub_image[:, :, 1]), np.mean(sub_image[:, :, 2])])

    for i in np.arange(radius, image.shape[0] - radius):
        for j in np.arange(radius, image.shape[1] - radius):
            result[i, j] = get_box_mean(i, j)
    return result[radius:-radius, radius:-radius].astype(np.uint8)


def method_3(image):
    result = np.zeros((image.shape[0] - (BOX_FILTER_SIZE - 1), image.shape[1] - (BOX_FILTER_SIZE - 1), 3))
    radius = BOX_FILTER_SIZE // 2
    indices = np.arange(-radius, radius + 1)
    for i in indices:
        for j in indices:
            result += image[i + radius:image.shape[0] + i - radius, j + radius:image.shape[1] + j - radius]
    return (result / (BOX_FILTER_SIZE ** 2)).astype(np.uint8)


im = cv.imread(FILE_PATH, cv.IMREAD_COLOR)

methods = [method_1, method_2, method_3]
for idx, func in enumerate(methods):
    start_time = time.time()
    res = func(im.copy())
    print(f'Method {idx + 1} -> time elapsed = {time.time() - start_time}')
    cv.imwrite(RES_PATHS[idx], res)
