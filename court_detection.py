# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:52:35 2026

Always click in this order USE DOUBLES LINES:
    1. Bottom-left (near court) 
    2. Bottom-right
    3. Top-left
    4. Top-right

@author: wyatt
"""

import cv2
import numpy as np

video_path = "data/video_files/BaselineElevatedView.mp4"

points = []

scale = 0.3  # make sure image is the right size


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scaled_x = int(x/scale)
        scaled_y = int(y/scale)
        points.append((scaled_x, scaled_y))
        print(f"Point added: {x}, {y}")

cap = cv2.VideoCapture(video_path)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.putText(resized_frame, f"Frame: {frame_number}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Scroll frames - press SPACE to select", resized_frame)

    key = cv2.waitKey(30)

    if key == ord(' '):  # SPACE = select this frame
        break
    elif key == ord('q'):  # quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# Now click points on selected frame
cv2.imshow("Click: BL, BR, TL, TR", resized_frame)
cv2.setMouseCallback("Click: BL, BR, TL, TR", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

court_points_pixel = np.array(points, dtype=np.float32)

court_points_real = np.array([
    [0, 0],
    [36, 0],
    [0, 78],
    [36, 78]
], dtype=np.float32)

H, _ = cv2.findHomography(court_points_pixel, court_points_real)

np.save("homography.npy", H)

print("✅ Homography saved!")