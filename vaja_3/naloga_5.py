import math
import numpy as np
from pathlib import Path
import ikpy.chain
from ikpy.chain import Chain
import ikpy.utils.plot as plot_utils
from ikpy.utils import geometry
from utils import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)

JOINT_NAMES = [
	'shoulder_pan',
	'shoulder_lift',
	'elbow_flex',
	'wrist_flex',
	'wrist_roll',
	'gripper',
]

def IK_demo(my_chain: Chain, pts: np.ndarray):
    global stopped

    stopped = False

    # stop with 'q' button
    def on_press(event):
        global stopped
        if event.key == "q":
            stopped = True

    target_orientation = geometry.rpy_matrix(0, np.deg2rad(180), 0)  # point down
    # target_orientation = geometry.rpy_matrix(0, np.deg2rad(90),0) # point forward

    radius = 0.5

    # N = 10
    fig, ax = plot_utils.init_3d_figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.mpl_connect("key_press_event", on_press)

    while not stopped:

        for target_position in pts:
            if stopped:
                break
            ax.cla()

            # ik = my_chain.inverse_kinematics(target_position, optimizer="scalar")  # ignores orientation
            ik = my_chain.inverse_kinematics(target_position, target_orientation, "all", optimizer='scalar') # includes orientation

            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(0, 2*radius)

            my_chain.plot(ik, ax, target=target_position)

            plt.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker=".", alpha=0.5)

            plt.draw()
            plt.pause(0.01)

def move_cubes(origin: np.ndarray, target: np.ndarray, cube_size=0.02):
    pass

def IK_control(robot, chain: Chain, pts: np.ndarray, repeat:bool=False):
    if repeat:
        while True:
            for pt in pts:
                ik = my_chain.inverse_kinematics(pt, optimizer='scalar')
                action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

                robot.send_action(action)

    for pt in pts:
        ik = my_chain.inverse_kinematics(pt, optimizer='scalar')
        action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

        robot.send_action(action)

if __name__ == "__main__":
    # Prepare robot
    URDF_PATH = "so101_new_calib.urdf"
    my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)
    my_chain.active_links_mask[0] = False
    # Configure robot
    port = "/dev/arm_f1"
    # robot_config = SO101FollowerConfig(port=port, id='arm_f1')
    calibration_dir = 'calibrations/'
    robot_config = SO101FollowerConfig(port=port, id='arm_f1', calibration_dir=Path(calibration_dir))
    robot = SO101Follower(robot_config)
    robot.connect()
    robot.bus.disable_torque()

    # IMPORTANT for setting maximum velocity and acceleration
    v = 500
    a = 10
    for j in JOINT_NAMES:
        robot.bus.write("Goal_Velocity", j, v)
        robot.bus.write("Acceleration", j, a)

    # Prepare DEMO
    offset = np.array([0.2, 0, 0.1])
    N_pts = 30
    pts = generate_figure_8(N=N_pts).T
    pts *= 0.15
    pts += offset
    IK_demo(my_chain, pts)

