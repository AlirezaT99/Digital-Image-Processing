import time

import cv2 as cv
import numpy as np

RESULT_DIR = 'results/'
RESULTS_PATH = ['res15.jpg', 'res16.jpg']
INPUTS_PATH = ['im03-holed.bmp', 'im04-holed.bmp']

PATCH_SIZES = [(37, 37, 3), (40, 40, 3)]
MERGE_WIDTH = 8


def insert_patch(image, patch, x, y):
    frame_x = min(image.shape[0] - x, patch.shape[0])
    frame_y = min(image.shape[1] - y, patch.shape[1])
    image[x:x + frame_x, y:y + frame_y] = patch[:frame_x, :frame_y]
    return image


def get_matching_block(image, template, valid_mask, mask=None):
    TOP_MATCHES = 3
    IGNORE_AROUND_RADIUS = 6

    def generate_mask(temp):
        new_temp = insert_patch(np.zeros(PATCH_SIZE[:2], dtype=np.uint8), temp, 0, 0)
        new_mask = np.zeros_like(new_temp)
        new_mask[:temp.shape[0], :temp.shape[1]] = 1
        return new_temp, new_mask

    def get_best_k_matches(values, k):
        best_matches = []
        min_score = np.min(values)
        while len(best_matches) < k:
            (i, j) = np.unravel_index(np.argmax(values), values.shape)
            values[
                max(0, i - IGNORE_AROUND_RADIUS):min(values.shape[0], i + IGNORE_AROUND_RADIUS),
                max(0, j - IGNORE_AROUND_RADIUS):min(values.shape[1], j + IGNORE_AROUND_RADIUS)
            ] = min_score
            if valid_mask[i:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]].all():
                best_matches.append((i, j))
        return best_matches

    im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    temp_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    if mask is None:
        temp_gray, mask = generate_mask(temp_gray)
    score = cv.matchTemplate(im_gray, temp_gray, cv.TM_CCORR_NORMED, mask=mask.astype(np.uint8))

    matches = get_best_k_matches(score, TOP_MATCHES)
    (x, y) = matches[np.random.randint(len(matches))]
    return image[x:x + PATCH_SIZE[0], y:y + PATCH_SIZE[1]]


def get_boundary(image_1, image_2, vertical):
    diff = np.power(cv.cvtColor(image_1, cv.COLOR_BGR2GRAY) - cv.cvtColor(image_2, cv.COLOR_BGR2GRAY), 2)
    if vertical:
        diff = diff.T

    dp = np.zeros_like(diff, dtype=np.int16)
    dp[:, 0] = diff[:, 0]

    path = np.zeros(diff.shape, dtype=np.int16)
    for j in range(1, dp.shape[1]):
        for i in range(dp.shape[0]):
            min_path = min([
                (-1, diff[i, j] + (dp[i - 1, j - 1] if i > 0 else np.inf)),
                (0, diff[i, j] + dp[i, j - 1]),
                (+1, diff[i, j] + (dp[i + 1, j - 1] if i < dp.shape[0] - 1 else np.inf))
            ], key=lambda tpl: tpl[1])
            dp[i, j], path[i, j] = min_path[1], min_path[0]

    mask = np.zeros(diff.shape, dtype=np.float64)
    min_col = np.argmin(dp[:, -1])
    for j in range(dp.shape[1] - 1, -1, -1):
        mask[:min_col, j] = 1
        mask[min_col, j] = .5
        min_col += path[min_col, j]

    return mask if not vertical else mask.T


def fill_holes(holed_image):
    def merge_overlaps_to_template(top, left):
        patch = np.zeros(PATCH_SIZE)
        patch = insert_patch(patch, top, 0, 0)
        patch = insert_patch(patch, left, 0, 0)
        patch_mask = np.ones(PATCH_SIZE[:2])
        patch_mask[top.shape[0]:, left.shape[1]:] = 0
        return patch.astype(np.uint8), patch_mask

    def merge_boundaries(boundary_v, boundary_h):
        merged_mask = np.zeros((boundary_v.shape[0], boundary_h.shape[1]))
        mask_v = insert_patch(merged_mask.copy(), boundary_v, 0, 0)
        mask_h = insert_patch(merged_mask.copy(), boundary_h, 0, 0)
        merged_mask = mask_h + mask_v
        merged_mask[merged_mask > 1] = 1
        return merged_mask

    filled_mask = np.sum(holed_image == [0, 0, 255], axis=2) < 3
    result = holed_image.copy()
    while True:
        if filled_mask.all():
            break
        red_pixels = np.argwhere(~filled_mask)
        i, j = red_pixels[0] - MERGE_WIDTH
        overlap_top = result[i:i + MERGE_WIDTH, j:j + PATCH_SIZE[1]]
        overlap_left = result[i:i + PATCH_SIZE[0], j:j + MERGE_WIDTH]
        overlap_right = result[i:i + PATCH_SIZE[0], j + PATCH_SIZE[1] - MERGE_WIDTH:j + PATCH_SIZE[1]].copy()
        overlap_bottom = result[i + PATCH_SIZE[0] - MERGE_WIDTH:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]].copy()

        is_right_filled = filled_mask[i:i + PATCH_SIZE[0], j + PATCH_SIZE[1] - MERGE_WIDTH:j + PATCH_SIZE[1]].all()
        is_bottom_filled = filled_mask[i + PATCH_SIZE[0] - MERGE_WIDTH:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]].all()

        template, mask = merge_overlaps_to_template(overlap_top, overlap_left)
        if is_right_filled:
            template[:, PATCH_SIZE[1] - MERGE_WIDTH:] = overlap_right
            mask[:, PATCH_SIZE[1] - MERGE_WIDTH:] = 1
        if is_bottom_filled:
            template[PATCH_SIZE[0] - MERGE_WIDTH:PATCH_SIZE[0], :] = overlap_bottom
            mask[PATCH_SIZE[0] - MERGE_WIDTH:, :] = 1
        block = get_matching_block(result, template, filled_mask, mask)
        block_overlap_top, block_overlap_left = block[:MERGE_WIDTH, :], block[:, :MERGE_WIDTH]
        boundary_mask_h = get_boundary(overlap_top, block_overlap_top, vertical=False)
        boundary_mask_v = get_boundary(overlap_left, block_overlap_left, vertical=True)
        boundary_mask = merge_boundaries(boundary_mask_v, boundary_mask_h)
        boundary_mask = cv.merge([boundary_mask, boundary_mask, boundary_mask])
        merged_top_left = boundary_mask * template + (1 - boundary_mask) * block

        result = insert_patch(result, merged_top_left, i, j)
        filled_mask[i:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]] = True

        mask_layer = boundary_mask[:, :, 0]
        block_overlap_bottom, block_overlap_right = block[-MERGE_WIDTH:, :], block[:, -MERGE_WIDTH:]
        if is_right_filled:
            boundary_mask_v = 1 - get_boundary(overlap_right, block_overlap_right, vertical=True)
            mask_layer += insert_patch(np.zeros_like(mask_layer), boundary_mask_v, 0, PATCH_SIZE[1] - MERGE_WIDTH)
            mask_layer[mask_layer > 1] = 1
            boundary_mask_v = mask_layer[:, -MERGE_WIDTH:]
            boundary_mask_v = cv.merge([boundary_mask_v, boundary_mask_v, boundary_mask_v])
            merged = (boundary_mask_v * overlap_right + (1 - boundary_mask_v) * block_overlap_right)
            result = insert_patch(result, merged.astype(np.uint8), i, j + PATCH_SIZE[1] - MERGE_WIDTH)
        if is_bottom_filled:
            boundary_mask_h = 1 - get_boundary(overlap_bottom, block_overlap_bottom, vertical=False)
            mask_layer += insert_patch(np.zeros_like(mask_layer), boundary_mask_h, PATCH_SIZE[0] - MERGE_WIDTH, 0)
            mask_layer[mask_layer > 1] = 1
            boundary_mask_h = mask_layer[-MERGE_WIDTH:, :]
            boundary_mask_h = cv.merge([boundary_mask_h, boundary_mask_h, boundary_mask_h])
            merged = (boundary_mask_h * overlap_bottom + (1 - boundary_mask_h) * block_overlap_bottom)
            result = insert_patch(result, merged.astype(np.uint8), i + PATCH_SIZE[0] - MERGE_WIDTH, j)
    return result


for idx, in_path in enumerate(INPUTS_PATH):
    im = cv.imread(in_path, cv.IMREAD_UNCHANGED)
    PATCH_SIZE = PATCH_SIZES[idx]
    start = time.time()
    res = fill_holes(im)
    cv.imwrite(f'{RESULT_DIR}{RESULTS_PATH[idx]}', res)
    print('time:', time.time() - start)
