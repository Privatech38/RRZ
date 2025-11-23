# a)
f = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]
k = [0.5, 1, 0.3]

result = [1.3, 1.8, 1.5, 0.71, 0.85, 0.91, 0.45, 0.1, 0.3, 1]

# b)
import numpy as np
import matplotlib.pyplot as plt

def simple_convolution(I, g):
    I = np.array(I)
    g = np.array(g)
    return np.array([np.dot(I[i:i+g.size], g) for i in range(0, I.size - g.size)])

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
plt.stairs(signal)
plt.stairs(kernel, edges=np.arange(N, len(kernel) + N + 1))
plt.stairs(rezultat, edges=np.arange(N, len(rezultat) + N + 1))
plt.show()

print(f'Moj rezultat: {rezultat}')

# c)
rezultat = np.convolve(rezultat, kernel)
plt.figure()
plt.stairs(signal)
plt.stairs(kernel, edges=np.arange(N, len(kernel) + N + 1))
plt.stairs(rezultat, edges=np.arange(N, len(rezultat) + N + 1))
plt.show()

print(f'Np rezultat: {result}')