import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import SimpleLineShadow, Normal, Stroke

# capture_f1.jpg koti
top_left = np.array([331, 321])
top_right = np.array([1312, 313])
bottom_left = np.array([414, 890])
bottom_right = np.array([1234, 889])

if __name__ == "__main__":
    # a)
    retval, mask = cv2.findHomography(np.array([top_left, top_right, bottom_left, bottom_right]),
                                      np.array([[0,0], [400, 0], [0, 270], [400, 270]]))

    original = cv2.imread('material/capture_f1.jpg', cv2.IMREAD_COLOR_RGB)
    warped = cv2.warpPerspective(original, retval, (400, 270))

    xx, yy = np.meshgrid(np.arange(0, 401, 10), np.arange(0, 271, 10))
    work_grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Draw corners
    plt.figure(figsize=(12.8, 9.6))
    plt.subplot(1,2,2)
    plt.imshow(warped)
    plt.scatter(work_grid_points[:, 0], work_grid_points[:, 1], c='b', marker='.', s=10)

    # b)
    inverse_H = np.linalg.inv(retval)
    homogeneous_work_grid = np.hstack((work_grid_points, np.ones((work_grid_points.shape[0], 1))))
    original_work_grid = (inverse_H@homogeneous_work_grid.T).T
    original_work_grid = original_work_grid[:, :2] / original_work_grid[:, [-1]]

    corners = np.array([original_work_grid[0], original_work_grid[40], original_work_grid[-41], original_work_grid[-1]]).astype(np.int32)
    work_corners = np.array([work_grid_points[0], work_grid_points[40], work_grid_points[-41], work_grid_points[-1]]).astype(np.int32)

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.scatter(original_work_grid[:, 0], original_work_grid[:, 1], c='b', marker='.', s=3)
    plt.scatter(corners[:, 0], corners[:, 1], c='r', marker='.', s=5)
    for i, (x, y) in enumerate(corners):
        text = plt.text(x, y, f"({work_corners[i][0]}, {work_corners[i][1]})", fontsize=10, color='white')
        # https://matplotlib.org/stable/users/explain/artists/patheffects_guide.html
        text.set_path_effects([Stroke(linewidth=2, foreground='black'), Normal()])
    plt.show()