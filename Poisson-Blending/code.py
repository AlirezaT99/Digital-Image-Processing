import time
import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

INPUT1_PATH = 'image03.png'
INPUT2_PATH = 'image04.png'
INPUT3_PATH = 'mask.bmp'
RESULTS_DIR = 'results/'
SOURCE_PATH = 'res05.jpg'
TARGET_PATH = 'res06.jpg'
RESULT_PATH = 'res07.jpg'


def normalize(image):
    if np.max(image) - np.min(image) > 255:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype(np.uint8)


def get_channels_laplacian(image):
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return cv.merge([
        convolve2d(image[:, :, 0], laplacian_filter, mode='same'),
        convolve2d(image[:, :, 1], laplacian_filter, mode='same'),
        convolve2d(image[:, :, 2], laplacian_filter, mode='same'),
    ])


def poisson_blending(source, target, mask):
    def get_mapping_laplacian(bin_mask):
        idx = 0
        res = np.zeros((bin_mask.shape[0], bin_mask.shape[1]), dtype=np.int32)
        laplacian = np.zeros((N, 3))
        for i in range(bin_mask.shape[0]):
            for j in range(bin_mask.shape[1]):
                if bin_mask[i, j] == 1:
                    laplacian[idx] = src_laplacian[i, j]
                    res[i, j] = idx
                    idx += 1
        return res, laplacian

    def create_sparse_equation(laplacian):
        sp_matrix = np.zeros((N, N))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    sp_matrix[mapping_N2[i, j], mapping_N2[i, j]] = 4
                    if mask[i, j - 1]:
                        sp_matrix[mapping_N2[i, j], mapping_N2[i, j - 1]] = -1
                    else:
                        laplacian[mapping_N2[i, j]] += target[i, j - 1]
                    if mask[i, j + 1]:
                        sp_matrix[mapping_N2[i, j], mapping_N2[i, j + 1]] = -1
                    else:
                        laplacian[mapping_N2[i, j]] += target[i, j + 1]
                    if mask[i - 1, j]:
                        sp_matrix[mapping_N2[i, j], mapping_N2[i - 1, j]] = -1
                    else:
                        laplacian[mapping_N2[i, j]] += target[i - 1, j]
                    if mask[i + 1, j]:
                        sp_matrix[mapping_N2[i, j], mapping_N2[i + 1, j]] = -1
                    else:
                        laplacian[mapping_N2[i, j]] += target[i + 1, j]
        return csc_matrix(sp_matrix), laplacian

    initial_mask = (mask == 255).astype(np.uint8)
    mask = cv.dilate(initial_mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), iterations=2)
    N = mask.sum()  # Num of variables
    src_laplacian = get_channels_laplacian(source)
    mapping_N2, laplacian_vector = get_mapping_laplacian(mask)
    A, b = create_sparse_equation(laplacian_vector)
    channels = []
    for channel in range(3):
        x = spsolve(A, b[:, channel])
        channels.append(normalize(x))
    target_colors = cv.merge(channels).reshape((-1, 3))
    new_source = np.zeros_like(target)
    new_source[mask == 1] = target_colors
    mask = cv.GaussianBlur(initial_mask, ksize=(5, 5), sigmaX=0, sigmaY=0)
    mask = np.repeat(np.expand_dims(initial_mask, axis=2), 3, axis=2)
    return (mask * new_source + (1 - mask) * target).astype(np.uint8)


im1 = cv.imread(INPUT1_PATH, cv.IMREAD_COLOR)
im2 = cv.imread(INPUT2_PATH, cv.IMREAD_COLOR)
msk = cv.imread(INPUT3_PATH, cv.IMREAD_GRAYSCALE)

start = time.time()
result = poisson_blending(im1, im2, msk)
print(f'time:', time.time() - start)

cv.imwrite(f'{RESULTS_DIR}{SOURCE_PATH}', im1)
cv.imwrite(f'{RESULTS_DIR}{TARGET_PATH}', im2)
cv.imwrite(f'{RESULTS_DIR}{RESULT_PATH}', result)
