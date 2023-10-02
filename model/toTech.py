import cv2
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path('/home/katya/agv_ws/src/traffic-light-yolov3-pkg/')    
VIDEO_PATH = str(PROJECT_ROOT /'test_video/person.mp4')
video = cv2.VideoCapture(VIDEO_PATH)

while (video.isOpened()):
    _, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(frame,frame, mask= mask)

    if np.any(mask):
        print('person detected')
    else:
        pass

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('SIGNS', frame)

