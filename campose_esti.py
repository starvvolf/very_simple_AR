# -*- coding: utf-8 -*-
#code referenced by: https://github.com/mint-lab/cv_tutorial

import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'kakao.mp4'
K = np.array([[490.05004599,   0.         ,658.52532342],
 [  0.,         488.98059273, 338.37840034],
 [  0.,           0.,           1.        ]]) # Derived from `calibrate_camera.py`
dist_coeff = np.array([ 0.01458104, -0.07937422,  0.00030928,  0.00335125,  0.12910111])
board_pattern = (6, 4)
board_cellsize = 0.017
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D box for simple AR
box_lower = board_cellsize * np.array([[3, 1,  0], [5, 1,  0], [5, 3,  0], [3, 3,  0]])
box_upper = board_cellsize * np.array([[3, 1, -2], [5, 1, -2], [5, 3, -2], [3, 3, -2]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])




# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        
        cv.polylines(img, [np.int32(line_lower)], True, (235, 175, 95), 3)
        
        moments = cv.moments(np.int32(line_upper))
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        center_point = (cx, cy)
        cv.circle(img, center_point, 2, (0, 0, 255), -1)
        
       
        
      
        for b in line_lower:
           cv.line(img, (int(b[0][0]), int(b[0][1])), center_point, (20, 80, 160), 3)
            
       

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()