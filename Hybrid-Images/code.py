import cv2 as cv
import numpy as np

INPUT1_PATH = 'Watson.jpg'
INPUT2_PATH = 'Radcliffe.jpg'
RESULTS_DIR = 'results/'

NEAR_ORIG_PATH = 'res19-near.jpg'
FAR_ORIG_PATH = 'res20-far.jpg'
NEAR_ADJUSTED_PATH = 'res21-near.jpg'
FAR_ADJUSTED_PATH = 'res22-far.jpg'
NEAR_DFT_PATH = 'res23-dft-near.jpg'
FAR_DFT_PATH = 'res24-dft-far.jpg'
HIGHPASS_PATH = f'res25-highpass-%s.jpg'
LOW_PASS_PATH = f'res26-lowpass-%s.jpg'
HIGH_PASSED_PATH = 'res27-highpassed.jpg'
LOW_PASSED_PATH = 'res28-lowpassed.jpg'
RESULT_PATH = 'res29-hybrid.jpg'
RES_NEAR_PATH = 'res30-hybrid-near.jpg'
RES_FAR_PATH = 'res31-hybrid-far.jpg'


def save_image(image, file_name):
    cv.imwrite(f'{RESULTS_DIR}{file_name}', image.astype(np.uint8))


def create_hybrid_image(near_im, far_im):
    RADIUS = 35

    def adjust_images(image_1, image_2):
        image_2 = cv.resize(image_2, image_1.shape[:2][::-1], interpolation=cv.INTER_CUBIC)
        eyes_1 = [[240, 155], [240, 284]]  # left, right
        eyes_2 = [[242, 164], [242, 281]]

        mid_eye_1 = (eyes_1[0][0] + eyes_1[1][0]) / 2, (eyes_1[0][1] + eyes_1[1][1]) / 2
        mid_eye_2 = (eyes_2[0][0] + eyes_2[1][0]) / 2, (eyes_2[0][1] + eyes_2[1][1]) / 2
        matrix = np.eye(3)
        matrix[0][2] = mid_eye_1[1] - mid_eye_2[1]
        matrix[1][2] = mid_eye_1[0] - mid_eye_2[0]
        image_2 = cv.warpPerspective(image_2, matrix, image_1.shape[:2][::-1])

        save_image(image_1, NEAR_ADJUSTED_PATH)
        save_image(image_2, FAR_ADJUSTED_PATH)
        return image_1, image_2

    def fourier(image):
        im_fft = np.empty(image.shape, dtype=complex)
        b, g, r = cv.split(image)
        im_fft[:, :, 0] = np.fft.fftshift(np.fft.fft2(b))
        im_fft[:, :, 1] = np.fft.fftshift(np.fft.fft2(g))
        im_fft[:, :, 2] = np.fft.fftshift(np.fft.fft2(r))
        return im_fft

    def inverse_fourier(image):
        im = np.empty(image.shape, dtype=complex)
        im[:, :, 0] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 0]))
        im[:, :, 1] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 1]))
        im[:, :, 2] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 2]))
        return np.real(im)

    def get_cut_off(shape, rad):
        result = np.zeros(shape)
        result = cv.circle(result, (shape[1] // 2, shape[0] // 2), int(1.3 * rad), (0.5, 0.5, 0.5), thickness=-1)
        result = cv.circle(result, (shape[1] // 2, shape[0] // 2), rad, (1, 1, 1), thickness=-1)
        return result

    def normalize_image(image):
        result = np.real(image)
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        return result.astype(np.uint8)

    near_im, far_im = adjust_images(near_im, far_im)

    cut_off = get_cut_off(near_im.shape[:2], RADIUS)
    save_image(normalize_image(1 - cut_off), HIGHPASS_PATH % 'none')
    save_image(normalize_image(cut_off), LOW_PASS_PATH % 'none')

    near_fft, far_fft = fourier(near_im), fourier(far_im)
    save_image(normalize_image(np.log(np.abs(near_fft))), NEAR_DFT_PATH)
    save_image(normalize_image(np.log(np.abs(far_fft))), FAR_DFT_PATH)
    far_fft[:, :, 0] = cut_off * far_fft[:, :, 0]
    far_fft[:, :, 1] = cut_off * far_fft[:, :, 1]
    far_fft[:, :, 2] = cut_off * far_fft[:, :, 2]
    near_fft[:, :, 0] = (1 - cut_off) * near_fft[:, :, 0]
    near_fft[:, :, 1] = (1 - cut_off) * near_fft[:, :, 1]
    near_fft[:, :, 2] = (1 - cut_off) * near_fft[:, :, 2]
    save_image(normalize_image(inverse_fourier(near_fft)), HIGH_PASSED_PATH)
    save_image(normalize_image(inverse_fourier(far_fft)), LOW_PASSED_PATH)

    mix = far_fft + near_fft
    hybrid_image = normalize_image(inverse_fourier(mix))
    save_image(hybrid_image, RESULT_PATH)

    shape = hybrid_image.shape[:2][::-1]
    save_image(cv.resize(hybrid_image, (shape[0] * 2, shape[1] * 2), interpolation=cv.INTER_CUBIC), RES_NEAR_PATH)
    save_image(cv.resize(hybrid_image, (shape[0] // 2, shape[1] // 2), interpolation=cv.INTER_AREA), RES_FAR_PATH)


im_1 = cv.imread(INPUT1_PATH, cv.IMREAD_COLOR)
im_2 = cv.imread(INPUT2_PATH, cv.IMREAD_COLOR)
create_hybrid_image(im_1, im_2)
