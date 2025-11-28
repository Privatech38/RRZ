import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def simple_gauss(sigma):
    absolute_sigma = 3 * abs(math.ceil(sigma))
    gauss_kernel = np.arange(-absolute_sigma, absolute_sigma + 1)
    gauss_kernel = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(gauss_kernel**2) / (2 * sigma**2))
    kernel_sum = np.sum(gauss_kernel)
    gauss_kernel /= kernel_sum
    return gauss_kernel

def gauss_2d(sigma):
    x_kernel = simple_gauss(sigma)
    return np.outer(x_kernel, x_kernel)

# a)
def simple_gaussdx(sigma):
    absolute_sigma = 3 * abs(math.ceil(sigma))
    gauss_kernel = np.arange(-absolute_sigma, absolute_sigma + 1)
    gauss_kernel = (-1 / (sigma ** 3 * math.sqrt(2 * math.pi))) * gauss_kernel * np.exp(-(gauss_kernel**2) / (2 * sigma**2))
    kernel_sum = np.sum(abs(gauss_kernel)) / 2
    gauss_kernel /= kernel_sum
    return gauss_kernel

def gaussdx_2d(sigma):
    x_kernel = simple_gaussdx(sigma)
    return np.outer(x_kernel, x_kernel)

def show_img(axs, img, title):
    axs.set_title(title)
    axs.imshow(img, cmap='gray')
    axs.axis('off')

# Slika
img = np.zeros((99, 99))
img[49, 49] = 1

SIGMA = 10

gauss_kernel = np.expand_dims(simple_gauss(SIGMA), 0)
gaussdx_kernel = np.expand_dims(simple_gaussdx(SIGMA), 0)


# Show image
figure, axs = plt.subplots(2, 3)

# I
show_img(axs[0, 0], img, 'I')
# I * G * G^T
dst = np.zeros_like(img)
cv2.filter2D(img, -1, gauss_kernel, dst)
cv2.filter2D(dst, -1, np.transpose(gauss_kernel), dst)
show_img(axs[1, 0], dst, 'I * G * G^T')

# I * G * D^T
dst = np.zeros_like(img)
cv2.filter2D(img, -1, gauss_kernel, dst)
cv2.filter2D(dst, -1, np.transpose(gaussdx_kernel), dst)
show_img(axs[0, 1], dst, 'I * G * D^T')

# I * D * G^T
dst = np.zeros_like(img)
cv2.filter2D(img, -1, gaussdx_kernel, dst)
cv2.filter2D(dst, -1, np.transpose(gauss_kernel), dst)
show_img(axs[1, 1], dst, 'I * D * G^T')

# I * G^T * D
dst = np.zeros_like(img)
cv2.filter2D(img, -1, gauss_kernel.T, dst)
cv2.filter2D(dst, -1, gaussdx_kernel, dst)
show_img(axs[0, 2], dst, 'I * G^T * D')

# I * D^T * G
dst = np.zeros_like(img)
cv2.filter2D(img, -1, gaussdx_kernel.T, dst)
cv2.filter2D(dst, -1, gauss_kernel, dst)
show_img(axs[1, 2], dst, 'I * D^T * G')

figure.show()

