import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import ikpy.utils.plot as plot_utils
from mpl_toolkits.mplot3d import Axes3D
from naloga_2 import stanford_manipulator, antropomorphic_manipulator

def plot(matrices: np.ndarray, ax: Axes3D):
    """
    Plot the DH parameter intermediate matrices.
    :param matrices: DH parameter intermediate matrices
    :param ax: 3D axes to draw on
    """
    matrices = np.insert(matrices, 0, np.eye(4), axis=0)
    points = np.array([T[:3, 3] for T in matrices])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c="r", marker=".", label="Joints")
    ax.plot(x, y, z, c="b", label="Joint connections")

def stanford_sliders():

    radius = 5

    # Initial parameter values
    initial_coeffs = [0.0, 0.0, 0.0]
    num_sliders = len(initial_coeffs)

    fig, ax = plot_utils.init_3d_figure()

    # Slider layout parameters
    slider_height = 0.03
    slider_spacing = 0.005
    slider_left = 0.15
    slider_width = 0.75
    slider_bottom_start = 0.05  # bottom position of the lowest slider

    slider_axes = []

    for i in range(num_sliders):
        # Position from bottom up
        bottom = slider_bottom_start + (num_sliders - i) * (slider_height + slider_spacing)
        ax_slider = plt.axes([slider_left, bottom, slider_width, slider_height])
        slider_axes.append(ax_slider)

    sliders = [
        Slider(
            ax=slider_axes[0],
            label="Joint 1 (Rotational)",
            valmin=-180.0,
            valmax=180.0,
            valinit=0
        ),
        Slider(
            ax=slider_axes[1],
            label="Joint 2 (Rotational)",
            valmin=-180.0,
            valmax=180.0,
            valinit=0
        ),
        Slider(
            ax=slider_axes[2],
            label="Joint 3 (Prismatic)",
            valmin=0.0,
            valmax=3.0,
            valinit=0.0
        )
    ]

    ax.cla()
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(0, 2*radius)

    # Create main figure and axes
    plt.subplots_adjust(bottom=0.45)  # Make space at the bottom for sliders

    # Plot the initial line
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stanford")

    plot(stanford_manipulator(initial_coeffs), ax)


    def update(val):
        """
        Update function for ALL sliders.
        This is called whenever any slider's value changes.
        """
        coeffs = [
            np.deg2rad(sliders[0].val),
            np.deg2rad(sliders[1].val),
            sliders[2].val
        ]
        ax.clear()

        # Update the plot
        plot(stanford_manipulator(coeffs), ax)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(0, 2*radius)

        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Connect all sliders to the same update function
    for s in sliders:
        s.on_changed(update)

    plt.show()

def antropomorphic_sliders():
    radius = 5

    # Initial parameter values
    initial_coeffs = [0.0, 0.0, 0.0]
    num_sliders = len(initial_coeffs)

    fig, ax = plot_utils.init_3d_figure()

    # Slider layout parameters
    slider_height = 0.03
    slider_spacing = 0.005
    slider_left = 0.15
    slider_width = 0.75
    slider_bottom_start = 0.05  # bottom position of the lowest slider

    slider_axes = []

    for i in range(num_sliders):
        # Position from bottom up
        bottom = slider_bottom_start + (num_sliders - i) * (slider_height + slider_spacing)
        ax_slider = plt.axes([slider_left, bottom, slider_width, slider_height])
        slider_axes.append(ax_slider)

    sliders = [
        Slider(
            ax=slider_axes[0],
            label="Joint 1 (Rotational)",
            valmin=-180.0,
            valmax=180.0,
            valinit=0
        ),
        Slider(
            ax=slider_axes[1],
            label="Joint 2 (Rotational)",
            valmin=-180.0,
            valmax=180.0,
            valinit=0
        ),
        Slider(
            ax=slider_axes[2],
            label="Joint 3 (Rotational)",
            valmin=-180.0,
            valmax=180.0,
            valinit=0
        )
    ]

    ax.cla()
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(0, 2*radius)

    # Create main figure and axes
    plt.subplots_adjust(bottom=0.45)  # Make space at the bottom for sliders

    # Plot the initial line
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Antropomorphic")

    plot(antropomorphic_manipulator(initial_coeffs), ax)


    def update(val):
        """
        Update function for ALL sliders.
        This is called whenever any slider's value changes.
        """
        coeffs = [np.deg2rad(s.val) for s in sliders]
        ax.clear()

        # Update the plot
        plot(antropomorphic_manipulator(coeffs), ax)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(0, 2*radius)

        # Redraw the figure canvas
        fig.canvas.draw_idle()

    # Connect all sliders to the same update function
    for s in sliders:
        s.on_changed(update)

    plt.show()


if __name__ == '__main__':
    stanford_sliders()
    antropomorphic_sliders()