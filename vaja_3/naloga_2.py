from typing import List
import numpy as np
from utils import dh_transform, show_coordinate_system
from math import pi
import matplotlib.pyplot as plt

# a)
#
#   d_i theta_i a_i alpha_i
# 1  0  theta_1  0   -pi/2
# 2 d_2 theta_2  0    pi/2
# 3 d_3    0     0     0

# b)
def stanford_manipulator(parameters: List[float]) -> np.ndarray:
    L1 = L2 = 5
    L3 = 2
    matrices = np.array([dh_transform(0, -np.pi/2, L1, parameters[0]),
                         dh_transform(0, np.pi/2, L2, parameters[1]),
                         dh_transform(0, 0, L3 + parameters[2], 0)])

    current_matrix = np.eye(4)
    results = []

    for m in matrices:
        current_matrix = current_matrix@m
        results.append(current_matrix)

    return np.array(results)


# print(stanford_manipulator([0.2, 0.4, 0.4]))

# c)
def antropomorphic_manipulator(parameters: List[float]) -> np.ndarray:
    length = 3

    matrices = np.array([
        dh_transform(0, pi/2, length, parameters[0]),
        dh_transform(length, 0, 0, parameters[1]),
        dh_transform(length, 0, 0, parameters[2]),
    ])

    current_matrix = np.eye(4)
    results = []

    for m in matrices:
        current_matrix = current_matrix@m
        results.append(current_matrix)

    return np.array(results)

# d)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
manipulator = stanford_manipulator([pi/4, pi/4, 1])
print(manipulator)
for matrix in manipulator:
    show_coordinate_system(ax, matrix, scale=0.6)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, 10])
plt.show()

# Antropomorfni
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
manipulator = antropomorphic_manipulator([pi/4, pi/4, -pi/4])
print(manipulator)
for matrix in manipulator:
    show_coordinate_system(ax, matrix, scale=0.6)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, 10])
plt.show()