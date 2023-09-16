import cv2
FindCam = []
for i in range(100):
    try:
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            FindCam.append(i)
    except:
        pass
print(str(FindCam))

# define a video capture object
vid = cv2.VideoCapture(8)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()