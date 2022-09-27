import time
import cv2 as cv
import numpy as np
from scipy.signal import medfilt2d
from skimage.segmentation import find_boundaries


RESULTS_DIR = 'results/'
INPUT_PATH = 'slic.jpg'
SEGMENTS = [64, 256, 1024, 2048]
RESULTS_PATH = ['res06.jpg', 'res07.jpg', 'res08.jpg', 'res09.jpg']

ALPHA = 0.02


def get_lab_edge(image):
    return np.abs(cv.Laplacian(image[:, :, 0], cv.CV_64F))


def slic(image, gradient, K):
    MAX_ITER = 50
    K = int(K ** 0.5) ** 2

    def draw_borders(labels_mat):
        return image * ~np.repeat(np.expand_dims(find_boundaries(labels_mat, mode='thick'), axis=2), 3, axis=2)

    def get_initial_centers(radius=2):  # makes for a 5x5 vicinity
        centers_in_row = int(K ** 0.5)
        initial_centers = []
        # Initialize centers
        center_dist = np.array(image.shape[:2]) // centers_in_row
        for i in range(centers_in_row):
            for j in range(centers_in_row):
                initial_centers.append([
                    center_dist[0] // 2 + i * center_dist[0],
                    center_dist[1] // 2 + j * center_dist[1]
                ])
        # Perturb to local minimum gradient
        perturbed_centers = np.zeros((K, 2))
        for c in range(K):
            center = initial_centers[c]
            window = gradient[center[0] - radius:center[0] + radius + 1, center[1] - radius:center[1] + radius + 1]
            min_grad = np.unravel_index(np.argmin(window), window.shape)
            perturbed_centers[c] = [center[0] - radius + min_grad[0], center[1] - radius + min_grad[1]]
        return perturbed_centers.astype(np.uint16)

    def convergence_reached(old, new, threshold=5):
        return (np.abs(new.astype(np.int16) - old.astype(np.int16)) < threshold).all()

    def enforce_connectivity(mat, kernel_size=25):
        return medfilt2d(mat.astype(np.float32), kernel_size).astype(np.uint32)

    labels = np.full(image.shape[:2], -1, dtype=np.uint32)
    energy = np.full(labels.shape, 1e10)
    S = np.max(image.shape[:2]) // int(K ** 0.5)
    centers = get_initial_centers()
    X, Y = np.indices(labels.shape)  # There's got to be a better way than next line!
    XY = np.array([X.flatten(), Y.flatten()]).T.reshape((image.shape[0], image.shape[1], 2))
    for t in range(MAX_ITER):
        for k in range(K):
            x_min, x_max = max(0, centers[k][0] - S), min(image.shape[0], centers[k][0] + S)
            y_min, y_max = max(0, centers[k][1] - S), min(image.shape[1], centers[k][1] + S)
            D_LAB = ((image[x_min:x_max, y_min:y_max] - image[centers[k][0], centers[k][1]]) ** 2).sum(axis=2)
            D_XY = ((XY[x_min:x_max, y_min:y_max] - centers[k]) ** 2).sum(axis=2)
            D = D_LAB + ALPHA * D_XY
            is_better_cluster = D < energy[x_min:x_max, y_min:y_max]
            np.putmask(energy[x_min:x_max, y_min:y_max], is_better_cluster, D)
            np.putmask(labels[x_min:x_max, y_min:y_max], is_better_cluster, k)

        old_centers = centers.copy()
        for k in range(K):
            points_k = XY[labels == k]
            if len(points_k) > 0:
                centers[k] = points_k.mean(axis=0).astype(np.uint16)

        if convergence_reached(old_centers, centers):
            break
    print('iterations:', t)
    labels = enforce_connectivity(labels)
    return draw_borders(labels)


im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
im_lab = cv.cvtColor(im, cv.COLOR_BGR2LAB)
im_gradient = get_lab_edge(im_lab)
for center_count, result_path in zip(SEGMENTS, RESULTS_PATH):
    start = time.time()
    clustered_im = slic(im, im_gradient, center_count)
    print(f'K={center_count} time:', time.time() - start)
    cv.imwrite(f'{RESULTS_DIR}{result_path}', clustered_im)
