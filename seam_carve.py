from numpy import zeros, dstack, roll
import numpy as np

def seam_carve(img, mode, mask=None):
    mode = mode.split()

    if mode[0] == 'vertical':
        img = np.transpose(img, axes=(1, 0, 2))
        if mask != None:
            mask = np.transpose(mask)

    (height, width) = img.shape[0:2]

    carve_mask = zeros((height, width))
    brightness = 0.299 * img[:, :, 0].astype(float) + 0.587 * img[:, :, 1].astype(float) + 0.114 * img[:, :, 2].astype(float)

    energy = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            dy = brightness[max(0, i - 1)][j] - brightness[min(height - 1, i + 1)][j]
            dx = brightness[i][max(0, j - 1)] - brightness[i][min(width - 1, j + 1)]
            energy[i][j] = (dx ** 2 + dy ** 2) ** 0.5
    if mask != None:
        energy += (256 * width * height) * mask

    for i in range(1, height):
        for j in range(width):
            min_step = energy[i - 1][j]
            if j != 0:
                min_step = min(min_step, energy[i - 1][j - 1])
            if j != width - 1:
                min_step = min(min_step, energy[i - 1][j + 1])
            energy[i][j] += min_step
    
    seam = 0
    for i in range(width):
        if energy[height - 1][i] < energy[height - 1][seam]:
            seam = i

    for i in range(height - 1, -1, -1):
        carve_mask[i][seam] = 1
        if i != 0:
            nextSeam = max(seam - 1, 0)
            if energy[i - 1][seam] < energy[i - 1][nextSeam]:
                nextSeam = seam
            if seam != width - 1 and energy[i - 1][seam + 1] < energy[i - 1][nextSeam]:
                nextSeam = seam + 1
            seam = nextSeam

    if mode[1] == 'shrink':
        img_mask = np.dstack((carve_mask.astype(bool),)*3)
        carve_img = img[np.invert(img_mask)]
        carve_img = np.reshape(carve_img, (height, -1, 3))
        if mask != None:
            mask = mask[np.invert(carve_mask.astype(bool))]
            mask = np.reshape(mask, (height, -1))
    else:
        carve_img = np.array([])
        for i in range(width):
            if carve_mask[0][i]:
                seam = i
        for i in range(height):
            level = img[i]
            pixel = []
            if seam == width - 1:
                pixel = level[seam]
            else:
                pixel = 0.5 * level[seam] + 0.5 * level[seam + 1]
                pixel = pixel.astype(np.uint8)
            level = np.insert(level, seam + 1, pixel, axis=0)
            carve_img = np.array([level]) if i == 0 else np.insert(carve_img, i, level, axis=0)
            if mask != None:
                mask_level = mask[i]
                mask_level = np.insert(mask_level, seam + 1, [0])
                mask_level[seam] = 1
                expand_mask = np.array([mask_level]) if i == 0 else np.insert(expand_mask, i, mask_level)

            if i != height - 1 and seam != width - 1 and carve_mask[i + 1][seam + 1]:
                seam = seam + 1
            elif i != height - 1 and seam != 0 and carve_mask[i + 1][seam - 1]:
                seam = seam - 1
        if mask != None:
            mask = expand_mask


    if mode[0] == 'vertical':
        carve_img = np.transpose(carve_img, axes=(1, 0, 2))
        carve_mask = np.transpose(carve_mask, axes=(1, 0))
        if mask != None:
            mask = np.transpose(mask)

    return (carve_img, mask, carve_mask)