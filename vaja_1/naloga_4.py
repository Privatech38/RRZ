import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cv2

# a)
rgb = np.array([255,34,126]) / 255
hsv = mcolors.rgb_to_hsv(rgb)
print(hsv)

# b)
hsv = np.array([0.65, 0.7, 0.15])
rgb = mcolors.hsv_to_rgb(hsv)
print(rgb * 255)

# c)
img = cv2.imread('RRZ_vaja1_material/trucks.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# RGB
plt.figure(figsize=(10,4))
plt.subplot(1,4,1)
plt.imshow(img_rgb)
plt.title('RGB')
plt.axis('off')
for i, col in enumerate(['R', 'G', 'B']):
    plt.subplot(1,4,i+2)
    plt.imshow(img_rgb[:,:,i], cmap='gray')
    plt.title(col)
    plt.axis('off')
plt.show()

# HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

plt.figure(figsize=(10,4))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV as RGB')
plt.axis('off')
for i, col in enumerate(['H', 'S', 'V']):
    plt.subplot(1,4,i+2)
    plt.imshow(img_hsv[:,:,i], cmap='gray')
    plt.title(col)
    plt.axis('off')
plt.show()

# d)
blue = img_rgb[:,:,2]
mask = blue > 150
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title('Blue > 150')
plt.axis('off')
plt.show()

# e)
rgb_float = img_rgb.astype(np.float32)
rgb_sum = np.sum(rgb_float, axis=2, keepdims=True)
rgb_sum[rgb_sum == 0] = 1
blue_norm = rgb_float[:,:,2] / rgb_sum[:,:,0]
mask_norm = blue_norm > 0.5
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(blue_norm, cmap='gray')
plt.title('Normalized Blue')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask_norm, cmap='gray')
plt.title('Norm. Blue > 0.5')
plt.axis('off')
plt.show()

# f)
lower = 80
upper = 130
hue = img_hsv[:,:,0]
print(img_hsv)
mask_blue = (hue >= lower) & (hue <= upper)
plt.figure()
plt.imshow(mask_blue, cmap='gray')
plt.title('Blue regions in HSV')
plt.axis('off')
plt.show()

# g)
def im_mask(img_rgb, mask):
    out = img_rgb.copy()
    out[~mask] = 0
    return out

masked_img = im_mask(img_rgb, mask_blue)
plt.imshow(masked_img)
plt.axis('off')
plt.title('Masked Image')
plt.show()
