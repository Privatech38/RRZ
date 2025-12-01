import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2

def draw_Hough_circles(img: numpy.ndarray, min_radius: int, max_radius: int, param1:int=50, param2:int=20, min_distance:int=20) -> numpy.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_distance, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    return img

plt.figure()
plt.imshow(draw_Hough_circles(cv2.imread("images/eclipse.jpg", cv2.IMREAD_COLOR_RGB), 45, 50))
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(draw_Hough_circles(cv2.imread("images/coins.jpg", cv2.IMREAD_COLOR_RGB), 85, 90))
plt.axis('off')
plt.show()

capture = cv2.VideoCapture('tf2_clip.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("Krogi", draw_Hough_circles(frame, 5, 60, 90, 45, 100))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()