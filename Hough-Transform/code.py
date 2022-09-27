import time
import cv2 as cv
import numpy as np

RESULT_DIR = 'results/'
INPUTS_PATH = ['im01.jpg', 'im02.jpg']
EDGES_PATHS = ['res01.jpg', 'res02.jpg']
HOUGH_PATHS = ['res03-hough-space.jpg', 'res04-hough-space.jpg']
LINES_PATHS = ['res05-lines.jpg', 'res06-lines.jpg']
CHESS_PATHS = ['res07-chess.jpg', 'res08-chess.jpg']
CORNERS_PATH = ['res09-corners.jpg', 'res10-corners.jpg']


def normalize_im(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image.astype(np.uint8)


def save_image(file_path, image):
    cv.imwrite(f'{RESULT_DIR}{file_path}', image)


def get_edge(image):
    return cv.Canny(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 500, 500)


def hough_transform(idx, image):
    THRESHOLD = 60
    LINES_MIN_DISTANCE = 5
    LINES_MIN_ANGLE = 5
    SHRINK = 3

    def remove_non_relevant_lines(acc):
        """ chess lines are perpendicular so there must be 2 thetas about 90 rows away (say between 75 and 105
        due to probable perspective). which indicate a considerable number of lines in the accumulator.
        Also chess borders need to be omitted.
        """
        row_sum = acc.sum(axis=1).reshape((-1, 1))
        lines = []
        while len(lines) < 2:
            max_row = np.argmax(row_sum)
            row_sum[max_row] = 0
            if len(lines) == 0 or (len(lines) == 1 and 75 < abs(lines[0] - max_row) < 105):
                lines.append(max_row)
        return lines

    def remove_outlier_lines(parallel_lines):
        LOW, HIGH = 0.75, 1.25

        def dist(x1, x2):
            return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)

        for t in parallel_lines.keys():
            lines = sorted(parallel_lines[t], key=lambda line: line[0][0])
            dist_vector = [dist(lines[line][0], lines[line + 1][0]) for line in range(len(lines) - 1)]
            median_distance = dist_vector[len(dist_vector) // 2]
            left = right = col = 0
            while dist_vector[col] < LOW * median_distance or dist_vector[col] > HIGH * median_distance:
                col += 1
                left += 1
            col, rev = 0, dist_vector[::-1]
            while rev[col] < LOW * median_distance or rev[col] > HIGH * median_distance:
                col += 1
                right += 1
            parallel_lines[t] = lines[left:len(lines) - right]
        return parallel_lines

    def draw_corners(background, lines):
        radius = 7
        red = (0, 0, 255)

        corners = []
        result = background.copy()
        lines_1, lines_2 = list(lines.values())[0], list(lines.values())[1]
        for line_1 in sorted(lines_1, key=lambda l: l[0][0]):  # y = ax + b
            a = (line_1[1][1] - line_1[0][1]) / (line_1[1][0] - line_1[0][0])
            b = line_1[0][1] - a * line_1[0][0]
            line_corners = []
            for line_2 in lines_2:  # y = cx + d
                c = (line_2[1][1] - line_2[0][1]) / (line_2[1][0] - line_2[0][0])
                d = line_2[0][1] - c * line_2[0][0]
                x_corner = int((d - b) / (a - c))
                y_corner = int(a * x_corner + b)
                line_corners.append((x_corner, y_corner))
            corners.append(sorted(line_corners, key=lambda tpl: tpl[0]))

        for line in corners:
            for corner in line:
                cv.circle(result, corner, radius, red, thickness=-1)
        return result

    def draw_lines(background, thetas, acc, precise=True):
        red = (0, 0, 255)
        LEN = 1000

        result = background.copy()
        lines = {}
        for t in thetas:
            t_s = [t] if precise else [t, t + 1, t - 1, t + 2, t - 2]
            theta_lines = []
            for tt in t_s:
                theta = tt * np.pi / 180
                while True:
                    if (acc[tt] == 0).all():
                        break
                    ro = np.argmax(acc[tt])

                    t_top, t_btm = max(0, tt - LINES_MIN_ANGLE), min(tt + LINES_MIN_ANGLE, acc.shape[0])
                    r_left, r_right = max(0, ro - LINES_MIN_DISTANCE), min(ro + LINES_MIN_DISTANCE, acc.shape[1])
                    acc[t_top:t_btm, r_left:r_right] = 0

                    ro -= acc.shape[1] // 2
                    ro *= SHRINK
                    x0, y0 = ro * np.cos(theta), ro * np.sin(theta)
                    half_w, half_h = result.shape[1] // 2, result.shape[0] // 2
                    (x1, y1) = int(x0 + LEN * -np.sin(theta)) + half_w, half_h - int(y0 + LEN * np.cos(theta))
                    (x2, y2) = int(x0 - LEN * -np.sin(theta)) + half_w, half_h - int(y0 - LEN * np.cos(theta))
                    theta_lines.append([(x0, y0), (x1, y1), (x2, y2)])
            lines[t] = theta_lines

        if not precise:
            lines = remove_outlier_lines(lines)
        for t in lines.keys():
            line = lines[t]
            for _, X1, X2 in line:
                cv.line(result, X1, X2, color=red, thickness=2)
            lines[t] = list(map(lambda tpl: tpl[1:], line))

        return result, lines

    orig_im = image.copy()
    image = cv.resize(image, (image.shape[1] // SHRINK, image.shape[0] // SHRINK), interpolation=cv.INTER_AREA)
    image = image[:image.shape[0] - (1 - image.shape[0] % 2), :image.shape[1] - (1 - image.shape[1] % 2)]
    half_width, half_height = image.shape[1] // 2, image.shape[0] // 2
    r_range, theta_accuracy = int(np.sqrt(half_width ** 2 + half_height ** 2)), 1

    edge = get_edge(image)
    save_image(EDGES_PATHS[idx], edge)
    accumulator = np.zeros((180, 2 * r_range))
    X, Y = np.where(edge == 255)
    for i, j in zip(X, Y):
        x, y = j - half_width, half_height - i
        for t in np.arange(0, 180, theta_accuracy):
            theta = t * np.pi / 180
            r = x * np.cos(theta) + y * np.sin(theta)
            accumulator[t, int(r) + r_range] += 1
    save_image(HOUGH_PATHS[idx], normalize_im(accumulator))

    accumulator[accumulator < THRESHOLD] = 0
    all_lines, _ = draw_lines(orig_im, range(180), accumulator.copy())
    save_image(LINES_PATHS[idx], all_lines)

    chess_lines_theta = remove_non_relevant_lines(accumulator)
    chess_image, chess_lines = draw_lines(orig_im, chess_lines_theta, accumulator.copy(), precise=False)
    save_image(CHESS_PATHS[idx], chess_image)

    chess_corners = draw_corners(orig_im, chess_lines)
    save_image(CORNERS_PATH[idx], chess_corners)


for index, input_path in enumerate(INPUTS_PATH):
    im = cv.imread(input_path, cv.IMREAD_COLOR)
    start = time.time()
    hough_transform(index, im)
    print(f'image{index}-time:', time.time() - start)
