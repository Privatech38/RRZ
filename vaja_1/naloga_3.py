import numpy as np
import matplotlib.pyplot as plt

picture = np.array([
    [5, 3, 2, 7, 1],
    [7, 1, 0, 0, 0],
    [4, 5, 7, 1, 1],
    [1, 3, 2, 1, 1],
    [5, 3, 1, 6, 3]
], dtype=np.uint8)

# a)
histogram = np.histogram(picture, bins=8, range=(0,8))
# print(histogram)

plt.figure()
plt.bar(range(8), histogram[0])
plt.xlabel('Nivo sivine')
plt.ylabel('Št. pojavitev')
plt.show()

# b)
cummulative_histogram = np.cumsum(histogram[0])
# print("Kumulativni histogram:", cummulative_histogram)

plt.figure()
plt.bar(range(8), cummulative_histogram)
plt.xlabel('Sivinski nivo')
plt.ylabel('Kumulativno')
plt.show()

# c)
histogram_4bit, _ = np.histogram(picture, bins=16, range=(0,16))

plt.figure()
plt.bar(range(16), histogram_4bit)
plt.xlabel('Nivo sivine')
plt.ylabel('Št. pojavitev')
plt.show()

# d)
import cv2
img = cv2.imread('RRZ_vaja1_material/umbrellas.jpg', 0)
hist_img, _ = np.histogram(img, bins=10)

plt.figure()
plt.bar(range(10), hist_img)
plt.xlabel('Nivo sivine')
plt.ylabel('Št. pojavitev')
plt.show()

# e)
def histogram_stretch(I, levels=256):
    I = I.astype(np.float32)
    vmin = np.min(I)
    vmax = np.max(I)
    out = (I - vmin) * (levels-1) / (vmax - vmin)
    out = np.clip(out, 0, levels-1)
    return out.astype(np.uint8)

img = cv2.imread('RRZ_vaja1_material/phone.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.hist(img.ravel(), bins=256, range=(0,256))
plt.title('Originalni histogram')
plt.show()

stretched = histogram_stretch(img)
plt.figure()
plt.imshow(stretched, cmap='gray')
plt.title('Slika po raztegu histograma')
plt.show()

plt.figure()
plt.hist(stretched.ravel(), bins=256, range=(0,256))
plt.title('Histogram po raztegu')
plt.show()

# f)
import cv2
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.title('Otsu pragovana slika')
plt.show()

# g)
# Otsu metoda deluje najboljše ko je velik kontrast med ozadjem in objekti