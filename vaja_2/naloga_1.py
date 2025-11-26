import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

# a)
f = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]
k = [0.5, 1, 0.3]

result = [1.3, 1.8, 1.5, 0.71, 0.85, 0.91, 0.45, 0.1, 0.3, 1]

# b)
def simple_convolution(I, g):
    I = np.array(I)
    g = np.array(g)
    return np.array([np.dot(I[i:i+g.size], g) for i in range(0, I.size - g.size + 1)])

signal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
          0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
kernel = [0.0022181959, 0.0087731348, 0.027023158, 0.064825185,
          0.12110939, 0.17621312, 0.19967563, 0.17621312, 0.12110939, 0.064825185,
          0.027023158, 0.0087731348, 0.0022181959]
rezultat = simple_convolution(signal, kernel)
# print(rezultat)

N = (len(kernel) - 1) // 2

plt.figure()
plt.title('simple_convolution()')
plt.stairs(signal)
plt.stairs(kernel, edges=np.arange(N, len(kernel) + N + 1))
plt.stairs(rezultat, edges=np.arange(N, len(rezultat) + N + 1))
plt.show()

# c)
rezultat = np.convolve(signal, kernel, mode='valid')
plt.figure()
plt.title('np.convolve()')
plt.stairs(signal)
plt.stairs(kernel, edges=np.arange(N, len(kernel) + N + 1))
plt.stairs(rezultat, edges=np.arange(N, len(rezultat) + N + 1))
plt.show()

# d)
def simple_gauss(sigma):
    absolute_sigma = 3 * abs(math.ceil(sigma))
    gauss_kernel = np.arange(-absolute_sigma, absolute_sigma + 1)
    gauss_kernel = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(gauss_kernel**2) / (2 * sigma**2))
    kernel_sum = np.sum(gauss_kernel)
    gauss_kernel /= kernel_sum
    return gauss_kernel

# e)
gauss_kernel = simple_gauss(2)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Podano jedro')
plt.plot(kernel)
plt.subplot(1, 2, 2)
plt.title('simple_gauss')
plt.plot(gauss_kernel)
plt.show()

plt.figure()
plt.plot(simple_gauss(0.5))
plt.plot(simple_gauss(1))
plt.plot(gauss_kernel)
plt.plot(simple_gauss(3))
plt.plot(simple_gauss(4))
plt.show()

# f)
plt.figure()
plt.imshow(plt.imread("konvolucija.png"))
plt.show()

# g)
kernel_2 = [0.1, 0.6, 0.4]
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(simple_convolution(simple_convolution(signal, gauss_kernel), kernel_2))
plt.subplot(1, 3, 2)
plt.plot(simple_convolution(simple_convolution(signal, kernel_2), gauss_kernel))
plt.subplot(1, 3, 3)
plt.plot(simple_convolution(signal, simple_convolution(gauss_kernel, kernel_2)))
plt.show()

# h)
def gauss_filter(img, sigma=1):
    filter_kernel = simple_gauss(sigma)
    # 1D
    dst_1d = np.zeros_like(img)
    cv2.filter2D(img, -1, filter_kernel, dst=dst_1d)
    column_filter = np.expand_dims(filter_kernel, axis=1)
    cv2.filter2D(dst_1d, -1, column_filter, dst=dst_1d)

    # 2D
    dst_2d = np.zeros_like(img)
    k_2d = filter_kernel @ column_filter
    cv2.filter2D(img, -1, k_2d, dst=dst_2d)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Filtracija 1D")
    plt.imshow(dst_1d, cmap="gray")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Filtracija 2D")
    plt.imshow(dst_2d, cmap="gray")
    plt.axis('off')
    plt.show()

gauss_filter(cv2.imread("images/lena_gauss.png", cv2.IMREAD_GRAYSCALE))
gauss_filter(cv2.imread("images/lena_sp.png", cv2.IMREAD_GRAYSCALE))

# i)
SHARPEN_KERNEL = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - (1.0 / 9.9) * np.ones((3, 3))
def sharpen_filter(img):
    dst = np.zeros_like(img)
    cv2.filter2D(img, -1, SHARPEN_KERNEL, dst=dst)
    return dst

img_sharpen = cv2.imread("images/fox.jpg", cv2.IMREAD_COLOR_RGB)
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_sharpen)
plt.title("Original")
plt.axis('off')
for i in range(1, 4):
    plt.subplot(2, 2, i + 1)
    img_sharpen = sharpen_filter(img_sharpen)
    plt.imshow(img_sharpen)
    plt.title(f"Sharpen {i}x")
    plt.axis('off')

plt.show()

# j)
x = np.concatenate((np.zeros(14), np.ones(11), np.zeros(15)))
# pokvarjeni signal
xc = np.copy(x)
xc[11] = 5
xc[18] = 5

def median_filter(I, sigma=2):
    window_size = sigma * 2 + 1
    center = window_size // 2
    return np.array([sorted(I[i:i + window_size])[center] for i in range(I.size - window_size + 1)])

plt.figure()
plt.plot(xc, label="Original")
plt.plot(np.pad(simple_convolution(x, simple_gauss(2)), 2, mode='edge'), label='Gauss')
plt.plot(np.pad(median_filter(x, sigma=3), 3, mode='edge'), label='Median')
plt.legend()
plt.show()

