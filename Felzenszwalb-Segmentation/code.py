import time
import cv2 as cv
import numpy as np
from skimage.segmentation import felzenszwalb, find_boundaries


RESULTS_DIR = 'results/'
INPUT_PATH = 'birds.jpg'
RESULT_PATH = 'res10.jpg'


def get_user_clicks():
    COLOR = (89, 82, 255)
    window_name = 'InitialPoints'
    points = []

    def handle_click(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append([y, x])
            shown_images.append(cv.circle(shown_images[-1].copy(), (x, y), radius=5, color=COLOR, thickness=2))
        elif event == cv.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                shown_images.pop()
        cv.imshow(window_name, shown_images[-1])

    cv.imshow(window_name, shown_images[-1])
    cv.setMouseCallback(window_name, handle_click)
    cv.waitKey(0)
    cv.destroyAllWindows()
    shown_images.clear()
    return np.array(points)


def draw_borders(image, labels_mat):
    return image * ~np.repeat(np.expand_dims(find_boundaries(labels_mat, mode='thick'), axis=2), 3, axis=2)


def apply_user_selection(labels, points):
    selected = np.zeros_like(im, dtype=bool)
    for point in points:
        selected += np.repeat(np.expand_dims(labels == labels[point[0], point[1]], axis=2), 3, axis=2)
    return im * selected


SHRINK = 3
im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
shown_images = [cv.resize(im, (im.shape[1] // SHRINK, im.shape[0] // SHRINK), interpolation=cv.INTER_AREA)]

preferred_points = get_user_clicks() * SHRINK

start = time.time()
segments = felzenszwalb(im, scale=400.0, sigma=1.1, min_size=40)
bordered_result = draw_borders(im, segments)

result = apply_user_selection(segments, preferred_points)

print(f'time:', time.time() - start)
cv.imwrite(f'{RESULTS_DIR}{RESULT_PATH}', result)
