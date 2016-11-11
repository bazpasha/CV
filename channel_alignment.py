import numpy as np
from numpy import array, dstack, roll
import skimage
from skimage.transform import rescale

def mse(image_a, image_b):
    image_c = image_a - image_b
    (width, height) = image_c.shape
    summa = np.linalg.norm(image_c)
    return summa / (width * height)

def cross(image_a, image_b):
    result = np.sum(image_a * image_b) / (np.sum(image_a) * np.sum(image_b))
    return -result

def min_shift(static, moving, max_shift=15):
    (height, width) = static.shape
    opt_shift = (0, 0)

    scale_value = 1
    while width * scale_value > 500:
        scale_value *= 0.5
    shift = max_shift

    while scale_value <= 1:
        cur_shift = (0, 0)
        opt_shift = (2 * opt_shift[0], 2 * opt_shift[1])
        min_value = 1000 * 1000 * 1000 * 1000

        static_image = rescale(static, scale_value)
        moving_image = rescale(moving, scale_value)
        scale_value *= 2

        moving_image = np.roll(moving_image, opt_shift[0] - shift - 1, axis=0)
        moving_image = np.roll(moving_image, opt_shift[1] - shift - 1, axis=1)
        for y in range(opt_shift[0] - shift, opt_shift[0] + shift + 1):
            moving_image = np.roll(moving_image, 1, axis=0)
            for x in range(opt_shift[1] - shift, opt_shift[1] + shift + 1):
                moving_image = np.roll(moving_image, 1, axis=1)

                left, right = max(0, x), min(width, width + x)
                top, bottom = max(0, y), min(height, height + y)

                metric_value = np.linalg.norm(static_image[top:bottom, left:right] - moving_image[top:bottom, left:right]) / ((right - left) * (bottom - top))
                if metric_value < min_value:
                    min_value = metric_value
                    cur_shift = (y, x)

            moving_image = np.roll(moving_image, -(2 * shift + 1), axis=1)

        opt_shift = cur_shift
        shift = 1

    return opt_shift

def align(bgr_image):
    (height, width) = bgr_image.shape
    bgr_image = skimage.img_as_ubyte(bgr_image)
    bgr_image = bgr_image[:, int(0.1 * width):int(0.9 * width)]
    channelHeight = height // 3
    blueChannel = bgr_image[int(0.1 * channelHeight) : int(0.9 * channelHeight), :]
    greenChannel = bgr_image[int(1.1 * channelHeight) : int(1.9 * channelHeight), :]
    redChannel = bgr_image[int(2.1 * channelHeight) : int(2.9 * channelHeight), :]

    red_shift = min_shift(greenChannel, redChannel)
    redChannel = np.roll(redChannel, red_shift[0], axis=0)
    redChannel = np.roll(redChannel, red_shift[1], axis=1)

    blue_shift = min_shift(greenChannel, blueChannel)
    blueChannel = np.roll(blueChannel, blue_shift[0], axis=0)
    blueChannel = np.roll(blueChannel, blue_shift[1], axis=1)

    result_image = dstack((redChannel, greenChannel, blueChannel))

    return result_image
