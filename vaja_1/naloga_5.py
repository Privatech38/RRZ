import cv2
import numpy as np
import matplotlib.pyplot as plt

# b)
img = cv2.imread("RRZ_vaja1_material/regions.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

n_labels, labels = cv2.connectedComponents(binary)
print("Število regij (vklj. z ozadjem):", n_labels)

plt.figure(figsize=(12,4))
for i in range(1, n_labels):
    plt.subplot(1, n_labels-1, i)
    plt.imshow(labels==i, cmap='gray')
    plt.title(f"Regija {i}")
    plt.axis('off')
plt.show()

# c)
import matplotlib.patches as patches

plt.figure()
plt.imshow(labels, cmap='nipy_spectral')
ax = plt.gca()

for i in range(1, n_labels):
    region_mask = (labels == i)
    y, x = np.where(region_mask)
    # Centroid
    cx, cy = x.mean(), y.mean()
    plt.scatter(cx, cy, c='red')
    # Bounding box
    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()
    rect = patches.Rectangle((minx, miny), maxx - minx, maxy - miny, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    plt.text(cx, cy, f"{i}", color="white", fontsize=12)
plt.title("Centroidi in pravokotniki")
plt.axis('off')
plt.show()

# d)
I = np.array([
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,1,0,0,0,1,0],
    [0,1,0,1,0,0,1,1,0],
    [1,1,1,1,0,1,0,0,0],
    [1,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0],
    [0,1,0,0,0,1,1,1,0]
], dtype=np.uint8)

k = np.array([
    [1,1,0],
    [1,1,1],
    [1,1,0]
], dtype=np.uint8)

er = cv2.erode(I, k)
dil = cv2.dilate(I, k)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(I)
plt.subplot(2, 2, 2)
plt.imshow(er)
plt.subplot(2, 2, 3)
plt.imshow(dil)
plt.show()

# e)
img = cv2.imread("RRZ_vaja1_material/regions_noise.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
n_labels, labels = cv2.connectedComponents(binary)
print("Število regij (vklj. ozadje):", n_labels)

for i in range(1, n_labels):
    area = np.sum(labels == i)
    print(f"Regija {i}: velikost = {area}")

# f)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

dilated = cv2.dilate(binary, kernel_rect)
eroded = cv2.erode(binary, kernel_rect)

plt.figure(figsize=(10,5))
for i, name in enumerate((binary, dilated, eroded)):
    plt.subplot(1,3,i+1)
    plt.imshow(name, cmap='gray')
    plt.axis('off')
plt.show()

# g)
# Manual opening (erode -> dilate)
opening = cv2.dilate(cv2.erode(binary, kernel_rect), kernel_rect)
# Manual closing (dilate -> erode)
closing = cv2.erode(cv2.dilate(binary, kernel_rect), kernel_rect)

# OpenCV built-in
opening_cv = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_rect)
closing_cv = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect)

plt.figure(figsize=(10,5))
for i, name in enumerate((opening, closing, opening_cv, closing_cv)):
    plt.subplot(2,2,i+1)
    plt.imshow(name, cmap='gray')
    plt.axis('off')
plt.show()

# Cleaned
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_rect)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_rect)

plt.figure(figsize=(10,5))
plt.title('Cleaned regions')
plt.imshow(cleaned, cmap='gray')
plt.axis('off')
plt.show()

# h)
img = cv2.imread('RRZ_vaja1_material/bird.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)  # Eksperimentiraj s pragom

# Morfološke operacije
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Izvorna slika")
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(binary, cmap='gray')
plt.title("Maska po pragu")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(mask, cmap='gray')
plt.title("Izboljšana maska")
plt.axis('off')
plt.show()

