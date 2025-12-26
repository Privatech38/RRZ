from typing import List

from utils import ik_newton_dh, generate_figure_8
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, dtype
from naloga_3 import plot
import ikpy.utils.plot as plot_utils

from utils import fk_dh


# a)
# Težava je v tem, da je vedno vsaj ena rešitev

# b)
def calculate_params(path: np.ndarray, dh_params: List[dict], q0: List[float], fixed:bool=False) -> np.ndarray:
    params = np.zeros_like(path)

    q_prev = q0.copy()
    for i, point in enumerate(path):
        params[i], _, _, _ = ik_newton_dh(point, q_prev, dh_params)
        if not fixed:
            q_prev = params[i]

    return params

stanford_dh_params = [
    {
        "joint_type": "R",
        "a": 0,
        "alpha": -pi/2,
        "d": 5,
        "theta_offset": 0
    },
    {
        "joint_type": "R",
        "a": 0,
        "alpha": pi / 2,
        "d": 5,
        "theta_offset": 0
    },
    {
        "joint_type": "P",
        "a": 0,
        "alpha": -pi/2,
        "d_offset": 2,
        "theta_offset": 0
    }
]

antropomorphic_dh_params = [
    {
        "joint_type": "r",
        "a": 0.0,
        "alpha": pi/2.0,
        "d": 3.0,
        "theta_offset": 0.0
    },
    {
        "joint_type": "r",
        "a": 5.0,
        "alpha": 0.0,
        "d": 0.0,
        "theta_offset": 0.0
    },
    {
        "joint_type": "r",
        "a": 5.0,
        "alpha": 0.0,
        "d": 0.0,
        "theta_offset": 0.0
    },
]

def show_animated(path: np.ndarray, params: np.ndarray, dh_params: List[dict]):
    global stopped

    stopped = False

    # stop with 'q' button
    def on_press(event):
        global stopped
        if event.key == "q":
            stopped = True

    # N = 10
    fig, ax = plot_utils.init_3d_figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.mpl_connect("key_press_event", on_press)

    radius = 5

    while not stopped:

        for target_parameter in params:
            if stopped:
                break
            ax.cla()

            _, matrices = fk_dh(target_parameter, dh_params)

            plot(matrices, ax)

            x,y,z = path[:, 0], path[:, 1], path[:, 2]
            ax.scatter(x, y, z, c="purple", marker=".", alpha=0.5)
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(0, 2*radius)

            plt.draw()
            plt.pause(0.05)

if __name__ == "__main__":
    points = generate_figure_8().T
    offset = np.array([6, 0, 2])
    points *= 2
    points += offset
    show_animated(points, calculate_params(points, stanford_dh_params, [0.0, 0.0, 0.0]), stanford_dh_params)
    show_animated(points, calculate_params(points, stanford_dh_params, [0.0, 0.0, 0.0], fixed=True), stanford_dh_params)
    show_animated(points, calculate_params(points, antropomorphic_dh_params, [0.0, 0.0, 0.0]), antropomorphic_dh_params)
    show_animated(points, calculate_params(points, antropomorphic_dh_params, [0.0, 0.0, 0.0], fixed=True), antropomorphic_dh_params)