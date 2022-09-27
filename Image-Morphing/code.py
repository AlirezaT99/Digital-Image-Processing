import os
import time
import shutil
import cv2 as cv
import numpy as np
from scipy.spatial import Delaunay

INPUT1_PATH = 'image01.png'
INPUT2_PATH = 'image02.png'
POINTS1_PATH = 'points1.txt'
POINTS2_PATH = 'points2.txt'
RESULTS_DIR = 'results/'
FRAMES_DIR = 'temp_morph_frames'
IMAGE1_PATH = 'res01.jpg'
IMAGE2_PATH = 'res02.jpg'
FRAME15_PATH = 'res03.jpg'
FRAME30_PATH = 'res04.jpg'
MORPH_PATH = 'morph.mp4'

READ_FROM_FILES = True
FRAMES = 45
FPS = 15


def get_points():
    global shown_images

    def read_points(points_path):
        with open(points_path, 'r') as f:
            lines = f.readlines()
        return np.array(list(map(lambda line: tuple(map(int, line.split())), lines)))

    def save_points(points, points_path):
        with open(points_path, 'w') as f:
            f.write('\n'.join([f'{point[0]} {point[1]}' for point in points]))

    if READ_FROM_FILES:
        return read_points(POINTS1_PATH), read_points(POINTS2_PATH)

    shown_images = [im1.copy()]
    p1 = get_user_clicks('image 1')
    shown_images = [im2.copy()]
    p2 = get_user_clicks('image 2')
    assert len(p1) == len(p2)

    save_points(p1, POINTS1_PATH)
    save_points(p2, POINTS2_PATH)
    return p1, p2


def get_user_clicks(name):
    window_name = f'Initial points for {name}'
    points = []

    def handle_click(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            points.append([y, x])
            shown_images.append(cv.circle(shown_images[-1].copy(), (x, y), radius=8, color=(55, 42, 12), thickness=2))
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


def save_video():
    os.system(f'ffmpeg -y -i {FRAMES_DIR}/frame-%d.jpg -r {FPS}'
              + f' -vcodec mpeg4 {RESULTS_DIR}{MORPH_PATH} -loglevel quiet')
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)


def save_frame(image):
    global idx

    if idx == 1 and os.path.isdir(FRAMES_DIR):  # remove old frames
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    if not os.path.isdir(FRAMES_DIR):  # create folder
        os.mkdir(FRAMES_DIR)

    cv.imwrite(f'{FRAMES_DIR}/frame-{idx}.jpg', image)
    cv.imwrite(f'{FRAMES_DIR}/frame-{2 * FRAMES - idx + 1}.jpg', image)  # to create loop
    idx += 1


def add_image_corners(points):
    row, col, _ = im1.shape
    corners = [[0, 0], [0, col - 1], [row - 1, 0], [row - 1, col - 1]]
    frame_points = [[0, col // 2], [row // 2, 0], [row - 1, col // 2], [row // 2, col - 1]]
    return np.vstack([points, np.vstack([corners, frame_points])])


def morph_images():
    def save_middle_frames(image):
        if t + 1 == 15:
            cv.imwrite(f'{RESULTS_DIR}{FRAME15_PATH}', image)
        if t + 1 == 30:
            cv.imwrite(f'{RESULTS_DIR}{FRAME30_PATH}', image)

    def invert_points(pts):
        return np.vstack((pts[:, 1], pts[:, 0])).T.astype(np.float32)

    save_frame(im1)
    triangles = Delaunay(points1).simplices
    for t in range(1, FRAMES):
        alpha = t / FRAMES
        morph_points = (1 - alpha) * points1 + alpha * points2
        frame = np.zeros(im1.shape, dtype=np.float64)
        accumulative_mask = np.zeros(im1.shape, dtype=np.uint8)
        for tri_idx in range(triangles.shape[0]):
            tri_image1 = points1[triangles[tri_idx]]
            tri_image2 = points2[triangles[tri_idx]]
            tri_morph = morph_points[triangles[tri_idx]]

            affine1 = cv.getAffineTransform(invert_points(tri_image1), invert_points(tri_morph))
            affine2 = cv.getAffineTransform(invert_points(tri_image2), invert_points(tri_morph))
            warp1 = cv.warpAffine(im1, affine1, im1.shape[:2][::-1], borderMode=cv.BORDER_REFLECT_101)
            warp2 = cv.warpAffine(im2, affine2, im1.shape[:2][::-1], borderMode=cv.BORDER_REFLECT_101)
            mask = cv.fillConvexPoly(np.zeros_like(im1), invert_points(tri_morph).astype(np.int32), (1, 1, 1))
            mask[mask == accumulative_mask] = 0
            frame += (1 - alpha) * warp1 * mask + alpha * warp2 * mask
            accumulative_mask += mask

        frame = frame.astype(np.uint8)
        save_frame(frame)
        save_middle_frames(frame)
    save_frame(im2)


idx, shown_images = 1, []
im1 = cv.imread(INPUT1_PATH, cv.IMREAD_COLOR)
im2 = cv.imread(INPUT2_PATH, cv.IMREAD_COLOR)
points1, points2 = get_points()
points1, points2 = add_image_corners(points1), add_image_corners(points2)

start = time.time()
morph_images()
print(f'time:', time.time() - start)
cv.imwrite(f'{RESULTS_DIR}{IMAGE1_PATH}', im1)
cv.imwrite(f'{RESULTS_DIR}{IMAGE2_PATH}', im2)
save_video()
