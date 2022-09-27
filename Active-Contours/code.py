import os
import time
import ffmpeg
import shutil
import cv2 as cv
import numpy as np


RESULTS_DIR = 'results/'
INPUT_PATH = 'tasbih.jpg'
RESULT_PATH = 'res11.jpg'
CONTOUR_PATH = 'contour.mp4'
CONTOURS_DIR = 'q5_temp_frames'

COLOR = (55, 42, 12)
ALPHA = 5  # Elasticity
GAMMA = 1000  # External
MU = 100  # Shape prior
FRAME_STEP = 4
D_DECAY = 0.98
WINDOW = (3, 3)


def save_video():
    ffmpeg \
        .input(f'./{CONTOURS_DIR}/frame-%d.jpg', framerate=5) \
        .output(f'{RESULTS_DIR}{CONTOUR_PATH}') \
        .overwrite_output().global_args('-loglevel', 'quiet').run()
    shutil.rmtree(CONTOURS_DIR, ignore_errors=True)


def draw_circles(image, centers):
    result = image.copy()
    for y, x in centers:
        cv.circle(result, (x, y), radius=8, color=COLOR, thickness=2)
    return result


def get_gradient(image):
    im_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return np.abs(cv.Laplacian(im_hsv, cv.CV_64F)).sum(axis=2)


def boost_gradient(gradient):
    gradient[gradient > np.max(gradient) / 5] *= 50
    return cv.GaussianBlur(gradient, (11, 11), 0)


def save_contour(image):
    global idx

    if idx == 1 and os.path.isdir(CONTOURS_DIR):  # remove old frames
        shutil.rmtree(CONTOURS_DIR, ignore_errors=True)
    if not os.path.isdir(CONTOURS_DIR):  # create folder
        os.mkdir(CONTOURS_DIR)

    cv.imwrite(f'{CONTOURS_DIR}/frame-{idx}.jpg', image)
    idx += 1


def get_user_clicks():
    window_name = 'InitialPoints'
    points = []

    def handle_click(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append([y, x])
            shown_images.append(cv.circle(shown_images[-1].copy(), (x, y), radius=8, color=COLOR, thickness=2))
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


def active_contours(initial_points, gradients):
    n = len(initial_points)
    D_bar = ((initial_points - np.vstack((initial_points[1:], initial_points[0]))) ** 2).sum() / n
    ITERATIONS = 300

    def E_total(points, d_bar, gamma=GAMMA, after_iter=50):
        STEP_IN = 4
        assert len(points) == 2
        apply_prior = t > after_iter

        def E_internal(alpha=ALPHA, mu=MU):
            E_elastic = alpha * (np.sum((points[1] - points[0]) ** 2) - d_bar) ** 2

            diff = STEP_IN * (points[1] - points[0])[::-1] * [1, -1]
            inner_dot = diff + (points[1] + points[0]) / 2
            E_prior = mu * np.sum((points[0] - inner_dot) ** 2)

            return E_elastic + apply_prior * E_prior

        def E_external():
            return -sum([gradients[p[0], p[1]] ** 2 for p in points])

        return gamma * E_external() + E_internal()

    def convergence_reached(old, new):
        return np.abs(old - new).sum() == 0

    X, Y = np.indices(WINDOW)
    XY = (np.array([X.flatten(), Y.flatten()]).T.reshape((WINDOW[0], WINDOW[1], 2)) - (WINDOW[0] // 2)).reshape((-1, 2))
    m = len(XY)
    current_points = initial_points.copy()
    save_contour(draw_circles(im, current_points))
    for t in range(ITERATIONS):
        D_bar *= D_DECAY
        paths = []
        for v0_state in range(m):  # fix v1
            v0 = current_points[0] + XY[v0_state]

            energy = np.zeros((len(XY), n - 1), dtype=np.float64)
            for mm in range(m):
                energy[mm, 0] = E_total(np.vstack((v0, current_points[1] + XY[mm])), D_bar)
            previous = np.zeros((len(XY), n - 1), dtype=np.uint8)  # between 0 and m-1
            previous[:, 0] = v0_state

            for v in range(1, n - 1):
                for mm in range(m):
                    prev_edges = []
                    for mmm in range(m):
                        pair = np.vstack((current_points[v] + XY[mmm], current_points[v + 1] + XY[mm]))
                        prev_edges.append((energy[mmm, v - 1] + E_total(pair, D_bar), mmm))
                    best = min(prev_edges, key=lambda tpl: tpl[0])
                    energy[mm, v] = best[0]
                    previous[mm, v] = best[1]

            for mm in range(m):
                energy[mm, -1] += E_total(np.vstack((current_points[-1] + XY[mm], v0)), D_bar)

            displacement = []
            mm = np.argmin(energy[:, -1])
            min_energy = energy[mm, -1]
            for i in range(n - 2, -1, -1):
                displacement = [current_points[i + 1] + XY[mm]] + displacement
                mm = previous[mm, i]
            displacement = [v0] + displacement
            paths.append((min_energy, displacement))

        best_path = np.array(min(paths, key=lambda tpl: tpl[0])[1])
        if convergence_reached(current_points, best_path):
            break
        current_points = best_path.copy()

        if t % FRAME_STEP == 0:
            save_contour(draw_circles(im, current_points))

    print('iterations:', t)
    result = draw_circles(im, current_points)
    save_contour(result)
    return result


idx = 1

im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
shown_images = [im.copy()]
initial_contour = get_user_clicks()

start = time.time()
im_gradient = boost_gradient(get_gradient(im))
final_contour = active_contours(initial_contour, im_gradient)
print(f'time:', time.time() - start)
cv.imwrite(f'{RESULTS_DIR}{RESULT_PATH}', final_contour)
save_video()
