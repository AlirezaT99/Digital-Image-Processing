import time

import cv2 as cv
import numpy as np

RESULT_DIR = 'results/'
RESULTS_PATH = ['res11.jpg', 'res12.jpg', 'res13.jpg', 'res14.jpg']
INPUTS_PATH = ['Textures/texture03.jpg', 'Textures/texture11.jpeg', 'my-texture01.png', 'my-texture02.jpg']

RESULT_SIZE = 2500
PATCH_SIZE = (150, 150, 3)
MERGE_WIDTH = 40


def plot_together(texture, result, file_path):
    output = 255 * np.ones((60 + result.shape[0], 90 + result.shape[1] + texture.shape[1], 3), dtype=np.uint8)
    (tex_x, tex_y) = (30, 30)
    (res_x, res_y) = (30, 60 + texture.shape[1])
    color = (0, 0, 0)
    output = insert_patch(output, texture, tex_x, tex_y)
    output = insert_patch(output, result, res_x, res_y)
    output = cv.rectangle(output, (tex_y, tex_x), (tex_y + texture.shape[1], tex_x + texture.shape[0]), color, 2)
    output = cv.rectangle(output, (res_y, res_x), (res_y + result.shape[1], res_x + result.shape[0]), color, 2)
    cv.putText(output, 'Texture', (tex_y, tex_x - 5), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv.putText(output, 'Result', (res_y, res_x - 5), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv.imwrite(f'{RESULT_DIR}{file_path}', output)


def insert_patch(image, patch, x, y):
    frame_x = min(image.shape[0] - x, patch.shape[0])
    frame_y = min(image.shape[1] - y, patch.shape[1])
    image[x:x + frame_x, y:y + frame_y] = patch[:frame_x, :frame_y]
    return image


def get_matching_block(image, template, mask=None):
    TOP_MATCHES = 7
    IGNORE_AROUND_RADIUS = 10

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


def synthesize_texture(texture):
    def get_random_patch(image):
        x = np.random.randint(image.shape[0] - PATCH_SIZE[0])
        y = np.random.randint(image.shape[1] - PATCH_SIZE[1])
        return image[x:x + PATCH_SIZE[0], y:y + PATCH_SIZE[1]]

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

    def move_pointers(target, x, y):
        y += PATCH_SIZE[1]
        if y >= target.shape[1]:
            y = 0
            x += PATCH_SIZE[0]
        return x, y

    min_size = int(np.ceil(
        (RESULT_SIZE - PATCH_SIZE[0]) / (PATCH_SIZE[0] - MERGE_WIDTH)
    )) * (PATCH_SIZE[0] - MERGE_WIDTH) + PATCH_SIZE[0]
    result = np.zeros((min_size, min_size, 3), dtype=np.uint8)
    i = j = 0

    initial_block = get_random_patch(texture)
    result = insert_patch(result, initial_block, i, j)
    i, j = move_pointers(result, i, j)
    while i < result.shape[1]:
        if i == 0:  # find vertical merge boundary
            j -= MERGE_WIDTH
            overlap_left = result[i:i + PATCH_SIZE[0], j:j + MERGE_WIDTH]
            block = get_matching_block(texture, overlap_left)
            overlap_right = block[:, :MERGE_WIDTH]
            boundary_mask = get_boundary(overlap_left, overlap_right, vertical=True)
            boundary_mask = cv.merge([boundary_mask, boundary_mask, boundary_mask])
            merged = (boundary_mask * overlap_left + (1 - boundary_mask) * overlap_right).astype(np.uint8)
            result = insert_patch(result, block, i, j)
            result = insert_patch(result, merged, i, j)
            i, j = move_pointers(result, i, j)
        elif j == 0:  # find horizontal boundary
            i -= MERGE_WIDTH
            overlap_top = result[i:i + MERGE_WIDTH, j:j + PATCH_SIZE[1]]
            block = get_matching_block(texture, overlap_top)
            overlap_bottom = block[:MERGE_WIDTH, :]
            boundary_mask = get_boundary(overlap_top, overlap_bottom, vertical=False)
            boundary_mask = cv.merge([boundary_mask, boundary_mask, boundary_mask])
            merged = (boundary_mask * overlap_top + (1 - boundary_mask) * overlap_bottom).astype(np.uint8)
            result = insert_patch(result, block, i, j)
            result = insert_patch(result, merged, i, j)
            i, j = move_pointers(result, i, j)
        else:  # gotta find both and merge results
            j -= MERGE_WIDTH
            overlap_top = result[i:i + MERGE_WIDTH, j:j + PATCH_SIZE[1]]
            overlap_left = result[i:i + PATCH_SIZE[0], j:j + MERGE_WIDTH]
            template, mask = merge_overlaps_to_template(overlap_top, overlap_left)
            block = get_matching_block(texture, template, mask)
            overlap_bottom, overlap_right = block[:MERGE_WIDTH, :], block[:, :MERGE_WIDTH]
            boundary_mask_h = get_boundary(overlap_top, overlap_bottom, vertical=False)
            boundary_mask_v = get_boundary(overlap_left, overlap_right, vertical=True)
            boundary_mask = merge_boundaries(boundary_mask_v, boundary_mask_h)
            boundary_mask = cv.merge([boundary_mask, boundary_mask, boundary_mask])
            merged = boundary_mask * template + (1 - boundary_mask) * block
            result = insert_patch(result, merged, i, j)
            i, j = move_pointers(result, i, j)
    return result[:RESULT_SIZE, :RESULT_SIZE]


for in_path, res_path in zip(INPUTS_PATH, RESULTS_PATH):
    im = cv.imread(in_path, cv.IMREAD_COLOR)
    start = time.time()
    res = synthesize_texture(im)
    print('time:', time.time() - start)
    plot_together(im, res, res_path)
