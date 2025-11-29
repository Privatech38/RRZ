import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

def simple_gauss(sigma):
    absolute_sigma = 3 * abs(math.ceil(sigma))
    gauss_kernel = np.arange(-absolute_sigma, absolute_sigma + 1)
    gauss_kernel = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(gauss_kernel**2) / (2 * sigma**2))
    kernel_sum = np.sum(gauss_kernel)
    gauss_kernel /= kernel_sum
    return gauss_kernel

def simple_gaussdx(sigma):
    absolute_sigma = 3 * abs(math.ceil(sigma))
    gauss_kernel = np.arange(-absolute_sigma, absolute_sigma + 1)
    gauss_kernel = (-1 / (sigma ** 3 * math.sqrt(2 * math.pi))) * gauss_kernel * np.exp(-(gauss_kernel**2) / (2 * sigma**2))
    kernel_sum = np.sum(abs(gauss_kernel)) / 2
    gauss_kernel /= kernel_sum
    return gauss_kernel

def gradient_magnitude(I):
    gauss_kernel =  np.expand_dims(simple_gauss(3), 0)
    gaussdx_kernel = np.expand_dims(simple_gaussdx(3), 0)

    I_x = np.zeros_like(I)
    cv2.filter2D(I, -1, gauss_kernel.T, I_x)
    cv2.filter2D(I_x, -1, gaussdx_kernel, I_x)

    I_y = np.zeros_like(I)
    cv2.filter2D(I, -1, gauss_kernel, I_y)
    cv2.filter2D(I_y, -1, gaussdx_kernel.T, I_y)
    # Magnitude
    I_mag = np.sqrt(I_x**2 + I_y**2)
    I_dir = np.arctan2(I_y, I_x)
    # Colored
    hue = (I_dir + np.pi) / (2 * np.pi)
    I_mag_norm = I_mag / I_mag.max()
    I_hsv = np.zeros((*hue.shape, 3))
    I_hsv[..., 0] = hue
    I_hsv[..., 1] = 1.0
    I_hsv[..., 2] = I_mag_norm
    return I, I_x, I_y, I_mag, I_dir, I_hsv

def show_edges(maps):
    names = ["I", "I_x", "I_y", "I_mag", "I_dir", "I_hsv"]
    plt.figure(dpi=200)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        if i == 5:
            plt.imshow(hsv_to_rgb(maps[i]))
        else:
            plt.imshow(maps[i], cmap="gray")
        plt.title(names[i])
        plt.axis('off')
    plt.show()

img = cv2.imread("images/museum.jpg", cv2.IMREAD_GRAYSCALE)
img = np.float32(img)
magnitudes = gradient_magnitude(img)
show_edges(magnitudes)

# b)
def edges_simple(I_mag, threshold):
    return I_mag > threshold

plt.figure()
thresholds = np.linspace(0.2, 0.7, 6)
I_mag = np.copy(magnitudes[3])
I_mag /= np.max(I_mag)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    treshold = thresholds[i]
    plt.title(f"t = {treshold}")
    plt.imshow(edges_simple(I_mag, treshold), cmap='gray')
    plt.axis('off')
plt.show()

# c)
capture = cv2.VideoCapture('tf2_clip.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(gray, 60, 120)
    cv2.imshow("frame", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# d)
capture = cv2.VideoCapture('tf2_clip.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
    corners = cv2.cornerHarris(gray, 2,3,0.02)
    corners = cv2.dilate(corners, None)
    frame[corners > 0.01 * corners.max()] = [0, 0, 255]
    cv2.imshow("Corners", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()