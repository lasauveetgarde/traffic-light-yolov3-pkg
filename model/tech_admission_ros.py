#!/home/katya/agv_ws/src/traffic-light-yolov3-pkg/.venv/bin/python 
import rospy
import rospkg
from std_msgs.msg import Int8
from scripts.models import *  
from utils.datasets import *
from utils.utils import *
import cv2 as cv

# cap = cv.VideoCapture('/home/katya/agv_ws/src/traffic-light-yolov3-pkg/test_video/person.mp4')
cap = cv.VideoCapture(2)

video_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def detect():
    while True:
        ##ROS SECTION##
        pub = rospy.Publisher('detectPerson', Int8, queue_size=10)
        result = 0
        rospy.init_node('PersonDetector', anonymous=True)
        rospack = rospkg.RosPack()
        ##ROS SECTION END##

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

        lower_blue = np.array([100,150,130])
        upper_blue = np.array([140,255,235])
        mask = cv.inRange(hsv_cropped, lower_blue, upper_blue)
        cropped = cv.bitwise_and(cropped,cropped, mask= mask)
        cv.imshow('frame cropped', cropped)

        if np.any(mask):
            putText_w=int(video_w/6)
            putText_h=int(video_h/4)
            frame = cv.putText(frame, 'PERSON DETECTED',(putText_w,putText_h), cv.FONT_HERSHEY_SIMPLEX, 2 , (0,0,255), 4, cv.LINE_AA)
            result=1
            rospy.loginfo(result)
            pub.publish(result)

        else:
            result=0
            rospy.loginfo(result)
            pub.publish(result)

        cv.imshow('frame roi', frame)

        # cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect()