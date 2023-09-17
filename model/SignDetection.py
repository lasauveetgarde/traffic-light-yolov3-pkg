from maddrive_adas.sign_det.composer import BasicSignsDetectorAndClassifier
from maddrive_adas.sign_det.base import (
    AbstractComposer, DetectedInstance,
    AbstractSignClassifier, AbstractSignDetector
)
from maddrive_adas.sign_det.classifier import EncoderBasedClassifier
from maddrive_adas.sign_det.detector import YoloV5Detector

import cv2
import datetime
import numpy as np
now = datetime.datetime.now
import pathlib
import os
PROJECT_ROOT = pathlib.Path('/home/katya/agv_ws/src/traffic-light-yolov3-pkg/')    
DETECTOR_ARCHIVE = PROJECT_ROOT /'detector_archive'
CLASSIFIER_ARCHIVE = PROJECT_ROOT /'encoder_archive'
SUBCLASSIFIER_ARCHIVE = PROJECT_ROOT /'subclassifier_3.24_3.25_archive'
SIGNS_ROOT='/home/katya/agv_ws/src/traffic-light-yolov3-pkg/model/signs/'

VIDEO_PATH = str(PROJECT_ROOT /'test_video/person1.avi')

c: AbstractSignClassifier = EncoderBasedClassifier(
    config_path=str(CLASSIFIER_ARCHIVE),
    path_to_subclassifier_3_24_and_3_25_config=str(SUBCLASSIFIER_ARCHIVE)
)

d: AbstractSignDetector = YoloV5Detector(
    config_path=str(DETECTOR_ARCHIVE)
)

composer: AbstractComposer = BasicSignsDetectorAndClassifier(
    classifier=c,
    detector=d
) 
from IPython.display import display,Image

video = cv2.VideoCapture(VIDEO_PATH)
# video = cv2.VideoCapture(0)

display_handle1=display(1, display_id=True)
# TODO: plot detector confidences

# const for plot
COLOR = (0, 255, 0)
DELTA = 15
i=1
areas=[]
lst_frame_src_array = []
signs_array = []
sign_detected = False

font = cv2.FONT_HERSHEY_SIMPLEX

while (video.isOpened()):
    # save initial timeq
    t0 = now()
    # get video frame
    ret1, frame = video.read()
    ret2, frame_src = video.read()

    ret, frame_sc = video.read()
    frame_sc = frame_src
    cimg = frame_sc
    hsv = cv2.cvtColor(frame_sc, cv2.COLOR_BGR2HSV)

    # conver BRG to RGB for model input and get composer predition
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    frame_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2RGB)   
    detected_instance, predicted_signs = composer.detect_and_classify(frame_src) 

    size = frame_sc.shape

    for idx, sign in enumerate(predicted_signs):
       
        COORD_ARR, conf = detected_instance.get_abs_roi(idx)
        frame_src_array = np.array([[[COORD_ARR[0],COORD_ARR[1]]],
                [[COORD_ARR[0],COORD_ARR[3]]],
                [[COORD_ARR[2],COORD_ARR[3]]],
                [[COORD_ARR[2],COORD_ARR[1]]]])

        lst_frame_src_array.append(frame_src_array)

        area=cv2.contourArea(frame_src_array)
        areas.append(area)
        signs_array.append(sign[0])
        max_area= max(areas)
        index_area = areas.index(max_area)

    if len(signs_array) == 0:
         pass
            # frame_src = cv2.putText(frame_src, 'NOTHING', 
            #     (100,200),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     4, (255, 255, 255),
            #     20, cv2.LINE_AA
            #         )

    elif max_area <= 1800:
        pass
       
    else:
        frame_src = cv2.drawContours(
                frame_src, lst_frame_src_array, index_area, COLOR, 3
            )
        if signs_array[index_area] == []:
            sign_detected = False
            pass
        else:
            str_index = str(signs_array[index_area])
            print(str_index, type(str_index))
            new_index = str_index.replace(".", "_", 2)
            new_index=new_index+".png"
            if str_index == '5.19.1':
                x=int(lst_frame_src_array[0][0][0][0])-500
                print(x)
                y=int(lst_frame_src_array[0][0][0][1])
                h = 400
                w = 500
                # frame_src = cv2.rectangle(frame_src, (x-300,y), (x,y+400), (255,0,0), 3)
                cropped_frame = frame_src[y:y+h, x:x+w]
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)   

                # cv2.imshow("cropped frame", cropped_frame)

                hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([100,150,0])
                upper_blue = np.array([140,255,255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                cropped_frame = cv2.bitwise_and(cropped_frame,cropped_frame, mask= mask)
                print(type(mask))
                if np.any(mask):
                    print('person detected!')
                    frame_src = cv2.putText(frame_src, 'PERSON DETECTED', (100,200), font, 5 , (255,0,0), 4, cv2.LINE_AA)
                # cv2.imshow("cropped blue frame", cropped_frame)
            else:
                pass

            if os.path.exists(SIGNS_ROOT+new_index):
                img = cv2.imread(SIGNS_ROOT+new_index, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                size = 300
                logo = cv2.resize(img, (size, size))
                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                roi = frame_src[-size-10:-10, -size-10:-10]
                roi[np.where(mask)] = 0
                roi += logo
            else:
                 pass
            
        # transform to 
        
    frame_src = cv2.cvtColor(frame_src, cv2.COLOR_RGB2BGR)
    frame_src=cv2.resize(frame_src,(640,480))
    cv2.imshow('1', frame_src)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    areas=[]
    lst_frame_src_array = []
    max_area = 0
    signs_array=[]


video.release()
cv2.destroyAllWindows()