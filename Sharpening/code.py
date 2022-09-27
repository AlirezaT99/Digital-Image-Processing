import cv2 as cv
import numpy as np

INPUT_PATH = 'flowers.blur.png'
RESULTS_DIR = 'results/'
RESULT_PATHS = ['res04.jpg', 'res07.jpg', 'res11.jpg', 'res14.jpg']


def save_image(image, file_name):
    cv.imwrite(f'{RESULTS_DIR}{file_name}', image.astype(np.uint8))


def get_gaussian_kernel(kernel_shape, sigma):
    def gaussian_2d(x, y, sig):
        return np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2))) / (2 * np.pi * (sig ** 2))

    kernel = np.zeros(kernel_shape)
    kernel_rad = (kernel_shape[0] // 2, kernel_shape[1] // 2)
    x_range = np.arange(-kernel_rad[0], kernel_rad[0] + kernel_rad[0] % 2)
    y_range = np.arange(-kernel_rad[1], kernel_rad[1] + kernel_rad[1] % 2)
    for i in x_range:
        for j in y_range:
            kernel[i + kernel_rad[0], j + kernel_rad[1]] = gaussian_2d(i, j, sigma)
    return kernel / np.sum(kernel)


def save_unsharp_mask(mask, file_path):
    mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 255).astype(np.uint8)
    save_image(mask, file_path)


def normalize_image(signed_image):
    signed_image[signed_image < 0] = 0
    signed_image[signed_image > 255] = 255
    return signed_image.astype(np.uint8)


def save_kernel(kernel, file_path):
    kernel = 255 * ((kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel)))
    save_image(kernel, file_path)


def fourier(image):
    im_fft = np.empty(image.shape, dtype=complex)
    b, g, r = cv.split(image)
    im_fft[:, :, 0] = np.fft.fft2(b)
    im_fft[:, :, 1] = np.fft.fft2(g)
    im_fft[:, :, 2] = np.fft.fft2(r)
    return im_fft


def shift_fourier(image):
    image[:, :, 0] = np.fft.fftshift(image[:, :, 0])
    image[:, :, 1] = np.fft.fftshift(image[:, :, 1])
    image[:, :, 2] = np.fft.fftshift(image[:, :, 2])
    return image


def inverse_fourier_shift(image):
    im_ifft = np.empty(image.shape, dtype=complex)
    im_ifft[:, :, 0] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 0]))
    im_ifft[:, :, 1] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 1]))
    im_ifft[:, :, 2] = np.fft.ifft2(np.fft.ifftshift(image[:, :, 2]))
    return np.real(im_ifft)


def inverse_fourier(image):
    im_ifft = np.empty(image.shape, dtype=complex)
    im_ifft[:, :, 0] = np.fft.ifft2(image[:, :, 0])
    im_ifft[:, :, 1] = np.fft.ifft2(image[:, :, 1])
    im_ifft[:, :, 2] = np.fft.ifft2(image[:, :, 2])
    return np.real(im_ifft)


def sharpen_image_a(image):
    GAUSSIAN_PATH = 'res01.jpg'
    SMOOTHED_PATH = 'res02.jpg'
    UNSHARP_PATH = 'res03.jpg'

    GAUSSIAN_SIZE = (11, 11)
    GAUSSIAN_STD = 3
    ALPHA = 2

    gaussian_kernel = get_gaussian_kernel(GAUSSIAN_SIZE, GAUSSIAN_STD)
    save_kernel(gaussian_kernel, GAUSSIAN_PATH)

    smoothed_image = cv.filter2D(image, -1, gaussian_kernel)
    save_image(smoothed_image, SMOOTHED_PATH)

    unsharp_mask = image.astype(np.int) - smoothed_image
    save_unsharp_mask(unsharp_mask, UNSHARP_PATH)

    image = image + ALPHA * unsharp_mask
    return normalize_image(image)


def sharpen_image_b(image):
    LAPLACIAN_PATH = 'res05.jpg'
    UNSHARP_PATH = 'res06.jpg'

    def save_laplacian(kernel, file_path):
        normalized_kernel = (255 * ((kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel)))).astype(np.uint8)
        save_image(normalized_kernel, file_path)

    GAUSSIAN_SIZE = (11, 11)
    GAUSSIAN_STD, IMPULSE_STD = 3, 0.5
    K = 1

    laplacian = get_gaussian_kernel(GAUSSIAN_SIZE, GAUSSIAN_STD) - get_gaussian_kernel(GAUSSIAN_SIZE, IMPULSE_STD)
    save_laplacian(laplacian, LAPLACIAN_PATH)

    unsharp_mask = cv.filter2D(image, -1, laplacian)
    save_unsharp_mask(unsharp_mask, UNSHARP_PATH)

    image = image.astype(np.int) - K * unsharp_mask
    return normalize_image(image)


def sharpen_image_c(image):
    FFT_AMPL_PATH = 'res08.jpg'
    HIGHPASS_PATH = 'res09.jpg'
    RES_FFT_PATH = 'res10.jpg'

    def normalize_log(log_ampl):
        result = np.real(log_ampl)
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
        return result.astype(np.uint8)

    def get_high_pass_filter(shape):
        sigma = min(shape) / 20
        return 1 - 500 * get_gaussian_kernel(shape, sigma)

    K = 0.25

    image_fft = fourier(image)
    shifted_image = shift_fourier(image_fft)

    log_amplitude_image = np.log(np.abs(shifted_image))
    save_image(normalize_log(log_amplitude_image), FFT_AMPL_PATH)

    HP = get_high_pass_filter(image_fft.shape[:2])
    save_kernel(HP, HIGHPASS_PATH)

    sharped_fft = np.empty(image_fft.shape, dtype=complex)
    sharped_fft[:, :, 0] = (1 + K * HP) * shifted_image[:, :, 0]
    sharped_fft[:, :, 1] = (1 + K * HP) * shifted_image[:, :, 1]
    sharped_fft[:, :, 2] = (1 + K * HP) * shifted_image[:, :, 2]

    save_image(normalize_log(np.log(np.abs(sharped_fft))), RES_FFT_PATH)

    image = inverse_fourier_shift(sharped_fft)
    return normalize_image(np.real(image))


def sharpen_image_d(image):
    LAPLACIAN_FFT_PATH = 'res12.jpg'
    UNSHARP_PATH = 'res13.jpg'

    def get_laplacian_fft(im_fft):
        result = np.empty(im_fft.shape, dtype=complex)
        for u in np.arange(result.shape[0]):
            for v in np.arange(result.shape[1]):
                result[u, v] = 4 * (np.pi ** 2) * (u ** 2 + v ** 2) * im_fft[u, v]
        return result

    def save_laplacian(mask, file_name):
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 255
        save_image(mask, file_name)

    K = 1e-8

    image_fft = fourier(image)
    laplacian_fft = get_laplacian_fft(image_fft)
    save_laplacian(np.abs(laplacian_fft), LAPLACIAN_FFT_PATH)

    unsharp = inverse_fourier(laplacian_fft)
    save_unsharp_mask(np.real(unsharp), UNSHARP_PATH)

    image = image + K * unsharp
    return normalize_image(np.real(image))


im = cv.imread(INPUT_PATH, cv.IMREAD_COLOR)
methods = [sharpen_image_a, sharpen_image_b, sharpen_image_c, sharpen_image_d]
for idx, func in enumerate(methods):
    sharp_im = func(im)
    save_image(sharp_im, RESULT_PATHS[idx])
