import numpy as np
import cv2 as cv

cap = cv.VideoCapture("/home/katya/adas_system/1.avi")
index = 6.2
str_index = str(index)

new_index = str_index.replace(".", "_", 1)
new_index=new_index+".png"
print(type(new_index))
img = cv.imread('/home/katya/adas_system/signs/'+new_index, cv.IMREAD_COLOR)
size = 200
logo = cv.resize(img, (size, size))
img2gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 1, 255, cv.THRESH_BINARY)
  
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    roi = frame[-size-10:-10, -size-10:-10]
    roi[np.where(mask)] = 0
    roi += logo
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()