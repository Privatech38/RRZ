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
from time import sleep
import warnings

warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)

# X se začne pri 12cm

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
            ik = my_chain.inverse_kinematics(target_position, target_orientation, "all",
                                             optimizer='scalar')  # includes orientation

            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(0, 2 * radius)

            my_chain.plot(ik, ax, target=target_position)

            plt.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker=".", alpha=0.5)

            plt.draw()
            plt.pause(0.01)


def move_cube(robot: SO101Follower, chain: Chain, origin: np.ndarray, destination: np.ndarray, cube_size=0.02) -> dict:
    # TODO Set gripper pos based on cube size
    origin = origin + np.array([-0.015, -0.005, 0])
    target_orientation_down = geometry.rpy_matrix(0, np.deg2rad(180), 0)
    target_orientation_forward = geometry.rpy_matrix(0, np.deg2rad(90), 0)
    # Grab cube instructions
    starting_point = origin + np.array([0, 0, 2 * cube_size])
    grab_cube_points = np.linspace(starting_point, origin, 6)
    grab_cube_instructions = []
    for point in grab_cube_points:
        ik = chain.inverse_kinematics(point, target_orientation_down, "all", optimizer='scalar')
        action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}
        action['gripper.pos'] = np.float64(35)
        grab_cube_instructions.append(action)
    grab_cube_instructions[len(grab_cube_instructions) - 1]['gripper.pos'] = np.float64(7.5)
    # Move cube to destination
    above_destination = np.copy(destination) + np.array([0, 0, cube_size])
    above_origin = np.copy(origin)
    above_origin[2] = above_destination[2]
    move_above_origin = np.linspace(origin, above_origin, 10)
    move_above_destination = np.linspace(above_origin, above_destination, 30)
    # Join points
    move_above_destination = np.concatenate((move_above_origin, move_above_destination), axis=0)
    move_cube_instructions = []
    for point in move_above_destination:
        ik = chain.inverse_kinematics(point, target_orientation_down, "all", optimizer='scalar')
        action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}
        action['gripper.pos'] = np.float64(7.5)
        move_cube_instructions.append(action)

    # Execute instructions
    for i in range(len(grab_cube_instructions) - 1):
        robot.send_action(grab_cube_instructions[i])
        sleep(0.5)
    robot.send_action(grab_cube_instructions[len(grab_cube_instructions) - 1])
    sleep(1.5)

    for instruction in move_cube_instructions:
        robot.send_action(instruction)
        sleep(0.1)

    # Lower the cube into position
    lower_ik = chain.inverse_kinematics(destination + np.array([0, 0, -0.007]), target_orientation_down, "all",
                                        optimizer='scalar')
    lower_action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(lower_ik[1:])}
    lower_action['gripper.pos'] = np.float64(7.5)
    robot.send_action(lower_action)
    sleep(0.75)

    # Release the cube
    lower_action['gripper.pos'] = np.float64(35)
    robot.send_action(lower_action)
    sleep(0.75)

    # Clear the gripper from the area
    clear_ik = chain.inverse_kinematics(destination + np.array([-0.04, 0, 3 * cube_size]), target_orientation_down,
                                        "all", optimizer='scalar')
    clear_action = {JOINT_NAMES[i] + '.pos': np.rad2deg(v) for i, v in enumerate(clear_ik[1:])}
    clear_action['gripper.pos'] = np.float64(35)
    robot.send_action(clear_action)
    sleep(0.5)


def IK_control(robot, chain: Chain, pts: np.ndarray, repeat: bool = False):
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
    port = "/dev/arm_f2"
    # robot_config = SO101FollowerConfig(port=port, id='arm_f1')
    calibration_dir = 'calibrations/'
    robot_config = SO101FollowerConfig(port=port, id='arm_f2', calibration_dir=Path(calibration_dir))
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
    N_pts = 30
    pts = generate_triangle(N=N_pts)
    offset = np.array([0.2, -0.05, 0.05])
    pts += offset
    IK_demo(my_chain, pts)
    IK_control(robot, my_chain, pts)
    move_cube(robot, my_chain, np.array([0.25, 0.115, 0.005]), np.array([0.20, 0.015, 0.005]))
    move_cube(robot, my_chain, np.array([0.275, -0.05, 0.005]), np.array([0.20, 0.015, 0.025]))
    move_cube(robot, my_chain, np.array([0.30, 0.04, 0.005]), np.array([0.20, 0.015, 0.045]))