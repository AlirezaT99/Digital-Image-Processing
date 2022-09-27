import time
import cv2 as cv
import numpy as np

INPUT1_PATH = 'image05.png'
INPUT2_PATH = 'image06.png'
RESULTS_DIR = 'results/'
IMAGE1_PATH = 'res08.jpg'
IMAGE2_PATH = 'res09.jpg'
RESULT_PATH = 'res10.jpg'


def normalize(image):
    if np.max(image) - np.min(image) > 255:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype(np.uint8)


def blend_images():
    levels = 5
    feathering_size = [0.03, 0.05, 0.1, 0.2, 0.4, 0.9]

    def create_laplacian_stack(image):
        stack = []
        kernel_size = np.array([1, 1])
        image_cp = image.copy().astype(np.float64)
        for i in range(levels):
            kernel_size = kernel_size * 2 + 1
            blurred = cv.GaussianBlur(image_cp, ksize=kernel_size, sigmaX=0, sigmaY=0)
            stack.append(image_cp - blurred)
            image_cp = blurred.copy()
        stack.append(image_cp)
        return stack

    def get_feathered_mask(shape, ratio):
        vector = np.zeros(shape[1])
        left_idx, feather_width = int(shape[1] * (1 - ratio) // 2), int(ratio * shape[1])
        vector[left_idx:left_idx + feather_width] = np.linspace(1, 0, num=feather_width)
        vector[:left_idx] = 1
        return np.repeat(np.expand_dims(np.repeat(np.expand_dims(vector, axis=0), shape[0], axis=0), axis=2), 3, axis=2)

    stack1, stack2 = create_laplacian_stack(im1), create_laplacian_stack(im2)
    blended_image = np.zeros(im1.shape, dtype=np.float64)
    for level in range(levels, -1, -1):
        mask = get_feathered_mask(im1.shape[:2], feathering_size[level])
        blended_image += stack1[level] * mask + stack2[level] * (1 - mask)
    return normalize(blended_image)


im1 = cv.imread(INPUT1_PATH, cv.IMREAD_COLOR)
im2 = cv.imread(INPUT2_PATH, cv.IMREAD_COLOR)

start = time.time()
result = blend_images()
print(f'time:', time.time() - start)

cv.imwrite(f'{RESULTS_DIR}{IMAGE1_PATH}', im1)
cv.imwrite(f'{RESULTS_DIR}{IMAGE2_PATH}', im2)
cv.imwrite(f'{RESULTS_DIR}{RESULT_PATH}', result)
