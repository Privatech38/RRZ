import cv2
import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.patheffects import Stroke, Normal
from matplotlib.pyplot import figure

from material.workspace_utils import get_workspace_corners, calculate_homography_mapping, workspace, workspace_height, workspace_width

if __name__ == "__main__":
    img_name = 'capture_f1.jpg'
    original_img = cv2.imread(f'material/{img_name}', cv2.IMREAD_COLOR_RGB)

    workspace_corners = get_workspace_corners(original_img)
    H1, H2 = calculate_homography_mapping(workspace_corners)

    # a)

    # b)

    figsize_ = rcParams["figure.figsize"]
    figsize_ = [value * 3 for value in figsize_]

    plt.figure(figsize=figsize_)

    # Original image
    plt.subplot(231)
    plt.imshow(original_img)
    plt.scatter(workspace_corners[:, 0], workspace_corners[:, 1], c='r', marker='.')
    for i, (x, y) in enumerate(workspace_corners):
        text = plt.text(x, y, f"({workspace[i][0]}, {workspace[i][1]})", fontsize=10, color='white')
        # https://matplotlib.org/stable/users/explain/artists/patheffects_guide.html
        text.set_path_effects([Stroke(linewidth=2, foreground='black'), Normal()])
    plt.title(img_name)

    # Workspace
    plt.subplot(232)
    warped_workspace = cv2.warpPerspective(original_img, H1, (workspace_width, workspace_height))
    plt.imshow(warped_workspace)
    plt.title('preslikava')

    manual_corners = np.array([[80, 120], [workspace_width - 80, 120], [80, workspace_height - 315], [workspace_width - 80, workspace_height - 315]])
    plt.scatter(manual_corners[:, 0], manual_corners[:, 1], c='r', marker='.')
    for x, y in manual_corners:
        text = plt.text(x, y, f"({x}, {y})", fontsize=10, color='white')
        text.set_path_effects([Stroke(linewidth=2, foreground='black'), Normal()])

    # Mask
    plt.subplot(233)
    workspace_mask = np.zeros(warped_workspace.shape[:2], dtype=np.uint8)
    cv2.rectangle(workspace_mask, manual_corners[0], manual_corners[3], 255, -1)
    plt.imshow(workspace_mask, cmap='gray')
    plt.title('maska delovne površine')

    # Masked image
    plt.subplot(234)
    plt.title('delovna površina')
    masked_workspace = cv2.bitwise_and(warped_workspace, warped_workspace, mask=workspace_mask)
    plt.imshow(masked_workspace)

    # Converted to HSV
    plt.subplot(235)
    hsv_workspace = cv2.cvtColor(masked_workspace, cv2.COLOR_RGB2HSV)
    tresholded_workspace = cv2.inRange(hsv_workspace, (172 // 2, 100, 1), (263 // 2, 255, 255))
    plt.imshow(tresholded_workspace, cmap='gray')
    plt.title('maska objektov')

    # Show
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(tresholded_workspace)
    plt.subplot(236)
    plt.title('regije')
    plt.imshow(labels, cmap='nipy_spectral')

    plt.show()

