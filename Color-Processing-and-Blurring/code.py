import cv2 as cv
import numpy as np

FILE_PATH = 'Flowers.jpg'
RES_PATH = 'results/res06.jpg'

PINK_LOW_H = 270 * (180 / 360)
PINK_HIGH_H = 330 * (180 / 360)
YELLOW_LOW_H = 40 * (180 / 360)
YELLOW_HIGH_H = 60 * (180 / 360)

BOX_BLUR_SIZE = 11


def map_color(x):
    if PINK_LOW_H <= x <= PINK_HIGH_H:
        return (x - PINK_LOW_H) / (PINK_HIGH_H - PINK_LOW_H) * (YELLOW_HIGH_H - YELLOW_LOW_H) + YELLOW_LOW_H
    return x


def blur_image(image, box_size=BOX_BLUR_SIZE):
    indices = np.arange(-(box_size // 2), box_size // 2 + 1)
    result = np.zeros(image.shape)
    for i in indices:
        for j in indices:
            result += shift_image(image, i, j)
    return (result // (box_size ** 2)).astype(np.uint8)


def remove_blurred_frame(image):
    frame = BOX_BLUR_SIZE // 2
    return image[frame:-frame, frame:-frame]


def shift_image(ch, dx, dy):
    ch = np.vstack((np.zeros((dx, ch.shape[1], 3)), ch)) if dx > 0 else np.vstack((ch, np.zeros((-dx, ch.shape[1], 3))))
    ch = np.hstack((np.zeros((ch.shape[0], dy, 3)), ch)) if dy > 0 else np.hstack((ch, np.zeros((ch.shape[0], -dy, 3))))
    ch = ch[:-dx, :, :] if dx > 0 else ch[-dx:, :, :]
    ch = ch[:, :-dy, :] if dy > 0 else ch[:, -dy:, :]
    return ch.astype(np.uint8)


im = cv.imread(FILE_PATH, cv.IMREAD_COLOR)
im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

h, s, v = cv.split(im_hsv)
pink_region = (PINK_LOW_H <= h) & (h <= PINK_HIGH_H)
h = np.vectorize(map_color)(h)
yellowed_im = cv.merge([h, s, v])
yellowed_im = cv.cvtColor(yellowed_im, cv.COLOR_HSV2BGR)

blurred_im = blur_image(yellowed_im)
blurred_im[pink_region] = yellowed_im[pink_region]
blurred_im = remove_blurred_frame(blurred_im)

cv.imwrite(RES_PATH, blurred_im)
