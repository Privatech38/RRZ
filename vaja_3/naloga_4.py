from typing import List

from utils import ik_newton_dh, generate_figure_8, dh_transform, end_effector_pos
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

def calculate_params_ccd(path: np.ndarray, dh_params: List[dict], q0: List[float], fixed:bool=False) -> np.ndarray:
    params = np.zeros_like(path)

    q_prev = q0.copy()
    for i, point in enumerate(path):
        params[i] = ik_ccd(point, q_prev, dh_params)
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

def ik_ccd(target_pos, q0, dh_params, max_iter=20):

    end_effector, inter_matrices = end_effector_pos(q0, dh_params)
    inter_matrices.insert(0, np.eye(4))
    matrices = []
    for i, p in enumerate(dh_params):
        if p["joint_type"] == "r":
            matrices.append(dh_transform(
                p["a"],
                p["alpha"],
                p["d"],
                p["theta_offset"] + q0[i],
            ))
        elif p["joint_type"] == "l":
            matrices.append(dh_transform(
                p["a"],
                p["alpha"],
                p["d_offset"] + q0[i],
                p["theta_offset"]
            ))

    q = np.array(q0, dtype=float)

    def update_inter_matrices(start_index: int):
        for k in range(start_index, len(matrices)):
            inter_matrices[k + 1] = inter_matrices[k] @ matrices[k]
        return inter_matrices[len(matrices)][:3, 3]

    for i in range(max_iter):
        q_i = np.copy(q)
        for j, p in reversed(list(enumerate(dh_params))):
            min_distance = np.linalg.norm(target_pos - end_effector)
            joint_type = p["joint_type"]
            if joint_type == "r":
                for value in np.arange(-pi, 0, 0.05, dtype=float)[::-1]:
                    matrices[j] = dh_transform(
                        p["a"],
                        p["alpha"],
                        p["d"],
                        p["theta_offset"] + q[j] + value
                    )
                    current_distance = np.linalg.norm(target_pos - update_inter_matrices(j))
                    if current_distance < min_distance:
                        q_i[j] = value + q[j]
                        min_distance = current_distance
                    if current_distance > min_distance:
                        break

                for value in np.arange(0, pi, 0.05, dtype=float):
                    matrices[j] = dh_transform(
                        p["a"],
                        p["alpha"],
                        p["d"],
                        p["theta_offset"] + q_i[j] + value
                    )
                    current_distance = np.linalg.norm(target_pos - update_inter_matrices(j))
                    if current_distance < min_distance:
                        q_i[j] = value + q[j]
                        min_distance = current_distance
                    if current_distance > min_distance:
                        break
            elif joint_type == "l":
                for value in np.arange(0, 10, 0.1, dtype=float):
                    matrices[j] = dh_transform(
                        p["a"],
                        p["alpha"],
                        p["d_offset"] + q[j] + value,
                        p["theta_offset"]
                    )
                    current_distance = np.linalg.norm(target_pos - update_inter_matrices(j))
                    if current_distance < min_distance:
                        q_i[j] = value + q[j]
                        min_distance = current_distance
                    if current_distance > min_distance:
                        break
        q = q_i
    return q




if __name__ == "__main__":
    points = generate_figure_8().T
    offset = np.array([6, 0, 2])
    points *= 2
    points += offset
    # show_animated(points, calculate_params(points, stanford_dh_params, [0.0, 0.0, 0.0]), stanford_dh_params)
    # show_animated(points, calculate_params(points, stanford_dh_params, [0.0, 0.0, 0.0], fixed=True), stanford_dh_params)
    # show_animated(points, calculate_params(points, antropomorphic_dh_params, [0.0, 0.0, 0.0]), antropomorphic_dh_params)
    # show_animated(points, calculate_params(points, antropomorphic_dh_params, [0.0, 0.0, 0.0], fixed=True), antropomorphic_dh_params)
    # d)
    ccd = calculate_params_ccd(points, antropomorphic_dh_params, [0.0, 0.0, 0.0])
    print(ccd)
    show_animated(points, ccd, antropomorphic_dh_params)