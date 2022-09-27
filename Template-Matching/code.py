import cv2 as cv
import numpy as np

INPUT_PATH = 'Greek-ship.jpg'
PATCH_PATH = 'patch.png'
RESULT_PATH = 'results/res15.jpg'

SHRINK = 5
SCORE_THRESHOLD = 110


def get_resized_edge(image):
    gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_im = cv.resize(gray_im, (image.shape[1] // SHRINK, image.shape[0] // SHRINK), interpolation=cv.INTER_AREA)
    return cv.Canny(image=gray_im, threshold1=100, threshold2=200)


def highlight_results(diff):
    RED = (0, 0, 255)
    ship = im_ship.copy()
    diff_image = cv.resize(diff, (diff.shape[1] * SHRINK, diff.shape[0] * SHRINK), interpolation=cv.INTER_CUBIC)
    diff_image = cv.filter2D(diff_image, -1, np.ones((11, 11), np.float32) / 121)
    diff_image = diff_image > SCORE_THRESHOLD
    while True:
        x, y = np.unravel_index(np.argmax(diff_image), diff_image.shape)
        if not diff_image[x, y]:
            break
        diff_image[
            x - im_patch.shape[0] // 2:x + im_patch.shape[0] // 2,
            y - im_patch.shape[1] // 2:y + im_patch.shape[1] // 2
        ] = False
        ship = cv.rectangle(ship, (y, x), (y + im_patch.shape[1], x + im_patch.shape[0]), color=RED, thickness=3)
    return ship


def match_template(image, patch):
    def normalize_matrix(matrix):
        res = matrix.copy()
        res[(np.isnan(res))] = 0
        res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
        return res

    image, patch = get_resized_edge(image), get_resized_edge(patch)
    x_range = image.shape[0] - patch.shape[0]
    y_range = image.shape[1] - patch.shape[1]

    diff_matrix = np.zeros((x_range, y_range))
    for i in np.arange(x_range):
        for j in np.arange(y_range):
            window = image[i:i + patch.shape[0], j:j + patch.shape[1]]
            diff_matrix[i, j] = np.sum((patch - np.average(patch)) * (window - np.average(window))) \
                / ((np.sum((patch - np.average(patch)) ** 2) * np.sum((window - np.average(window)) ** 2)) ** 0.5)

    diff_im = normalize_matrix(diff_matrix)
    return highlight_results(diff_im)


im_ship = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
im_patch = cv.imread(PATCH_PATH, cv.IMREAD_COLOR)
result = match_template(im_ship, im_patch)
cv.imwrite(RESULT_PATH, result)
