import numpy as np
import cv2 as cv


cap = cv.VideoCapture('/home/katya/agv_ws/src/traffic-light-yolov3-pkg/test_video/person.mp4')
video_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:

    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   

    x= 300
    y= 260
    h = 400
    w = 700

    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = frame[y:y+h, x:x+w]
    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)   

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   

    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    mask = cv.inRange(hsv_cropped, lower_blue, upper_blue)
    cropped = cv.bitwise_and(cropped,cropped, mask= mask)

    if np.any(mask):
        putText_w=int(video_w/6)
        putText_h=int(video_h/4)
        frame = cv.putText(frame, 'PERSON DETECTED',(putText_w,putText_h), cv.FONT_HERSHEY_SIMPLEX, 2 , (0,0,255), 4, cv.LINE_AA)
        print('person detected')
    else:
        pass

    cv.imshow('frame roi', frame)

    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()