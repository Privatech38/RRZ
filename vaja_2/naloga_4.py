# Resolucija akumulatorskega polja:
import math

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

def accumulation(img: numpy.ndarray) -> numpy.ndarray:
    bins_theta = 300
    bins_rho = 300
    max_rho = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
    val_theta = np.linspace(-90, 90, bins_theta) / 180 * np.pi # vrednosti theta
    val_rho = np.linspace(-max_rho, max_rho, bins_rho)
    A = np.zeros((bins_rho, bins_theta))
    # Primer za točko (50, 90)
    img = img > 0.6
    for x, y in np.argwhere(img):
        rho = x * np.cos(val_theta) + y * np.sin(val_theta) # Izračunamo rho za vse vrednosti theta
        bin_rho = np.round((rho + max_rho) / (2 * max_rho) * len(val_rho))
        for i in range(bins_theta):
            if bin_rho[i] >= 0 and bin_rho[i] <= bins_rho - 1:
                A[int(bin_rho[i]), i] += 1
    return A

plt.figure()
img = cv2.imread('images/oneline.png', cv2.IMREAD_GRAYSCALE)
img = np.uint8(img)
plt.subplot(1,2,1)
plt.title("oneline.png")
plt.imshow(accumulation(cv2.Canny(img, 100, 200)), cmap='gnuplot2')
plt.subplot(1,2,2)
img = np.uint8(cv2.imread('images/rectangle.png', cv2.IMREAD_GRAYSCALE))
plt.title("rectangle.png")
plt.imshow(accumulation(cv2.Canny(img, 100, 200)), cmap='gnuplot2')
plt.show()

def draw_lines(img: numpy.ndarray) -> numpy.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    gray = cv2.Canny(gray, 100, 200)
    linesP = cv2.HoughLinesP(gray, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv2.LINE_AA)
    return img

plt.figure()
img = cv2.imread('images/building.jpg', cv2.IMREAD_COLOR_RGB)
plt.imshow(draw_lines(img))
plt.axis('off')
plt.show()

plt.figure()
img = cv2.imread('images/pier.jpg', cv2.IMREAD_COLOR_RGB)

plt.imshow(draw_lines(img))
plt.axis('off')
plt.show()

capture = cv2.VideoCapture('tf2_clip.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("Črte", draw_lines(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()