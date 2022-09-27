import cv2 as cv
import numpy as np

INPUT_PATH = 'books.jpg'
RESULTS_DIR = 'results/'

# Order:    1   3
#           2   4
image_corners = {
    'res16.jpg': np.array([[209, 666], [104, 382], [394, 603], [289, 318]]),
    'res17.jpg': np.array([[741, 358], [465, 405], [709, 157], [430, 208]]),
    'res18.jpg': np.array([[968, 813], [665, 624], [1099, 610], [795, 422]]),
}


def get_destination_corners(shape):
    return np.array([[0, 0], [shape[0] - 1, 0], [0, shape[1] - 1], [shape[0] - 1, shape[1] - 1]])


def get_result_size(corners):
    top_left, bottom_left, top_right = corners[:3]
    width = np.sqrt((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2)
    height = np.sqrt((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2)

    return int(height), int(width), 3


def inverse_warp(image, matrix, shape):
    def weighted_average(float_x, float_y):
        int_x, int_y = int(float_x), int(float_y)
        a = float_x - int_x
        b = float_y - int_y

        return (1 - b) * ((1 - a) * image[int_x][int_y] + a * image[int_x + 1][int_y]) \
            + b * ((1 - a) * image[int_x][int_y + 1] + a * image[int_x + 1][int_y + 1])

    result = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vector = matrix @ np.array([i, j, 1])
            source_x, source_y = (vector / vector[-1])[:2]
            result[i, j] = weighted_average(source_x, source_y)
    return result.astype(np.uint8)


def warp_image(image, corners):
    result_shape = get_result_size(corners)
    dest_corners = get_destination_corners(result_shape)
    homography_matrix, _ = cv.findHomography(dest_corners, corners)
    print(homography_matrix)

    return inverse_warp(image, homography_matrix, result_shape)


im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
for file_name, book_corners in image_corners.items():
    warped_image = warp_image(im, book_corners)
    cv.imwrite(f'{RESULTS_DIR}{file_name}', warped_image)
