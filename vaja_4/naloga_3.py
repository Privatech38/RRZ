import cv2
import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.patheffects import Stroke, Normal
from matplotlib.pyplot import figure

from material.workspace_utils import get_workspace_corners, calculate_homography_mapping, workspace, workspace_height, workspace_width

if __name__ == '__main__':
    w, h = 1600, 1200
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    calibration = np.load('material/wide_calibration_data.npz')

    print(calibration)

    points_in_robot_space = np.array([[0.43, 0.21, 1], [0.43, -0.21, 1], [0.15, 0.21, 1], [0.15, -0.21, 1]])

    plt.figure()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.undistort(frame, calibration['camera_matrix'], calibration['dist_coeffs'])
        workspace_corners = get_workspace_corners(frame)
        H1, H2 = calculate_homography_mapping(workspace_corners)

        inverse_H1 = np.linalg.inv(H1)
        inverse_H2 = np.linalg.inv(H2)

        for x, y in workspace_corners:
            cv2.circle(frame, (x, y), 3, (0,0,255))

        # a)
        manual_pts = (inverse_H1 @ inverse_H2 @ points_in_robot_space.T).T
        manual_pts = manual_pts[:, :2] / manual_pts[:, [-1]]

        for x, y in manual_pts.astype(np.int32):
            print(x, y)
            cv2.circle(frame, (x, y), 3, (0,0,255))


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

