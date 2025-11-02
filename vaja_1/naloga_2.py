import numpy as np
import matplotlib.pyplot as plt

proj_fn = lambda f, p:  (-f*p[0]/p[2], -f*p[1]/p[2])

# a)
tree = [0, 5, 14]
print(f'Preslikano drevo je visoko {proj_fn(0.1, tree)[1]}m')

# b)
displacement_fn = lambda a, t: a * t**2 / 2
seconds = np.linspace(0, 30, 300)
ACCELERATION = 0.5
distances = (2.5, 0, 10 + displacement_fn(ACCELERATION, seconds))
widths = np.abs(proj_fn(0.1, distances)[0]) * 100

plt.plot(seconds, widths)
plt.xlabel('Čas v sekundah')
plt.ylabel('Širina v centimetrih')
plt.show()

# c)
# Slabosti:
# - Slaba svetloba
# - Ni mogoče določati goriščne razdalje

# d)
pixel_convert_fn = lambda pixel_size, dpi: 0.0254 / dpi * pixel_size

image_size = pixel_convert_fn(200, 2500)
actual_size = -1 * image_size * 95 / 0.06
print(f'Actual size: {actual_size}')


