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
from naloga_1 import