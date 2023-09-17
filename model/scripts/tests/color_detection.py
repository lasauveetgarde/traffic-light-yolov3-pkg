import cv2
from cv2 import RETR_EXTERNAL
from cv2 import CHAIN_APPROX_SIMPLE
import numpy as np
cap = cv2.VideoCapture('/home/katya/agv_ws/src/traffic-light-yolov3-pkg/test_video/person1.avi')

while(1):		
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(res, contours, 0, (0, 255, 0), 2)

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            area = cv2.contourArea(c)
            if area >= 15:
                start_point=(cX-100,cY+100)
                end_point=(cX+100,cY-100)
                color=(0,0,255)
                thickness = 5
                res = cv2.rectangle(res, start_point, end_point, color, thickness)
    frame=cv2.resize(frame,(640,480))
    res=cv2.resize(res,(640,480))
    cv2.imshow('frame',frame)
    cv2.imshow('res',res)

    # This displays the frame, mask
    # and res which we created in 3 separate windows.
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Destroys all of the HighGUI windows.
cv2.destroyAllWindows()

# release the captured frame
cap.release()
