import time
import cv2 as cv
import numpy as np

RESULTS_DIR = 'results/'
INPUT_PATH = 'park.jpg'
RESULT_PATH = 'res05.jpg'

WINDOW_SIZE = 3
SAME_COLOR_DIFF = np.array([40, 20, 20])
CONVERGE_THR = np.array([2, 1, 1])  # trade-off between speed and accuracy
SHRINK = 3

new_cluster_number = 0


def mean_shift(image):
    clusters = dict()

    def colorize_result(labels):
        bgr_image = cv.cvtColor(image.astype(np.uint8), cv.COLOR_Luv2BGR)
        colored = np.zeros_like(bgr_image, dtype=np.uint8)
        for k in clusters.keys():
            colored[labels == k] = bgr_image[labels == k].mean(axis=0)
        return colored

    def get_cluster_number(color):
        global new_cluster_number

        def almost_same_color(luv1, luv2):
            return (np.abs(luv1 - luv2) < SAME_COLOR_DIFF).all()

        for k, v in clusters.items():
            if almost_same_color(v, color):
                return k

        clusters[new_cluster_number] = color.astype(np.uint8)
        new_cluster_number += 1
        return new_cluster_number - 1

    completed_mask = np.full(image.shape[:2], False)
    result = np.zeros(completed_mask.shape)
    image = cv.cvtColor(image, cv.COLOR_BGR2Luv).astype(np.int16)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not completed_mask[i, j]:
                center = image[i, j]
                while True:
                    condition = (image > center - WINDOW_SIZE).all(axis=2) & (image < center + WINDOW_SIZE).all(axis=2)
                    sphere = image[condition]
                    new_center = sphere.mean(axis=0)
                    if (np.abs(new_center - center) < CONVERGE_THR).all():
                        break
                    center = new_center
                cluster_id = get_cluster_number(new_center)
                # assign this cluster to same colors
                color_location = np.array(image > image[i, j] - SAME_COLOR_DIFF).all(axis=2) \
                                 & np.array(image < image[i, j] + SAME_COLOR_DIFF).all(axis=2)
                result[color_location] = cluster_id
                completed_mask[color_location] = True
                # assign this cluster to colors around local maxima
                color_location = np.array(image > center - SAME_COLOR_DIFF).all(axis=2) \
                                 & np.array(image < center + SAME_COLOR_DIFF).all(axis=2)
                result[color_location] = cluster_id
                completed_mask[color_location] = True
    print('clusters count:', len(clusters.keys()))
    return colorize_result(result)


im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
im = cv.resize(im, (im.shape[1] // SHRINK, im.shape[0] // SHRINK), interpolation=cv.INTER_AREA)
start = time.time()
segmented_im = mean_shift(im)
segmented_im = cv.resize(segmented_im, im.shape[:2][::-1], interpolation=cv.INTER_CUBIC)
print(f'time:', time.time() - start)
cv.imwrite(f'{RESULTS_DIR}{RESULT_PATH}', segmented_im)
