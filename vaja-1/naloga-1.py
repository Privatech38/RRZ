import cv2
import matplotlib.pyplot as plt
import numpy as np

ime_slike = './vaja-1/RRZ_vaja1_material/bird.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

print(f'{im.shape=}')
print(f'{im.dtype=}')
plt.clf()

def gray_scale_image(image):
    gray_image = np.mean(image, axis=2, dtype=np.float32).astype(np.float32)
    return gray_image

def cut_out_image(image, x1, y1, x2, y2):
    cut_image = image[y1:y2, x1:x2, :]
    return cut_image

def negate_image(image):
    neg_image = 255 - image
    return neg_image

def treshold(image, t):
    return (np.where(image > t, image, 0)).astype(np.float32)

plt.imshow(treshold(gray_scale_image(im), 120), cmap='gray')
plt.show()