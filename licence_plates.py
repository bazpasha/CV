import numpy as np
from skimage.io import imread, imsave
from os import walk
import os.path
from skimage.filters import threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk, square
from skimage.transform import resize
from skimage.measure import label, regionprops
from itertools import combinations
import skimage

def cross_correlation(a, b):
    (height, width) = a.shape
    result = np.linalg.norm(a - b)
    result /= height * width
    return result

def generate_template(digit_dir_path):
    template_size = (42, 42)
    template = np.zeros(template_size, dtype=float)
    test_number = 0
    for root, _, files in walk(digit_dir_path):
        for file in files:
            digit = imread(os.path.join(root, file))
            digit = resize(digit, template_size)
            thresh = threshold_otsu(digit)
            template += digit > thresh
            test_number += 1
    template *= 1 / test_number
    return template

def detect_number(img, digit_templates):
    template_size = (42, 42)
    digit = resize(img, template_size)
    thresh = threshold_otsu(digit)
    digit = (digit > thresh).astype(float)
    result = 0
    min_cor = cross_correlation(digit, digit_templates[0])
    for i in range(10):
        if cross_correlation(digit, digit_templates[i]) < min_cor:
            min_cor = cross_correlation(digit, digit_templates[i])
            result = i
    return result


def recognize(img, digit_templates):
    number = skimage.exposure.adjust_gamma(img, 1.55)
    number = median(number, disk(1))
    thresh = threshold_otsu(number)

    binary = (number < thresh).astype(float)
    labeled = label(binary)
    regions = [(i.bbox[1], i.bbox[0], i.bbox[3], i.bbox[2]) for i in regionprops(labeled) if i.area > 100]


    answer = ()
    value = 100
    for a, b, c in combinations(sorted(regions), 3):
        wa, wb, wc = a[2] - a[0], b[2] - b[0], c[2] - c[0]
        ha, hb, hc = a[3] - a[1], b[3] - b[1], c[3] - c[1]
        cur_value = (max(wa, wb, wc) - min(wa, wb, wc)) + max(ha, hb, hc) - min(ha, hb, hc)
        cur_value += (b[0] - a[2]) + (c[0] - b[2])
        cur_value -= min(ha, hb, hc)/2
        if cur_value < value:
            answer = (a, b, c)
            value = cur_value
    answer = [detect_number(img[r[1]:r[3], r[0]:r[2]], digit_templates) for r in answer]
    return tuple(answer)