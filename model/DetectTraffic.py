import argparse

from scripts.models import *  
from utils.datasets import *
from utils.utils import *

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

PROJECT_ROOT = pathlib.Path('/home/blackwidow/catkin_ws/src/traffic-light-yolov3-pkg/')    
DETECTOR_ARCHIVE = PROJECT_ROOT /'detector_archive'
CLASSIFIER_ARCHIVE = PROJECT_ROOT /'encoder_archive'
SUBCLASSIFIER_ARCHIVE = PROJECT_ROOT /'subclassifier_3.24_3.25_archive'
SIGNS_ROOT= PROJECT_ROOT /'model/signs/'

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

# video = cv2.VideoCapture(VIDEO_PATH)
video = cv2.VideoCapture(8)
video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

display_handle1=display(1, display_id=True)

COLOR = (255, 0, 0)
DELTA = 15
i=1
areas=[]
lst_frame_src_array = []
signs_array = []
sign_detected = False

font = cv2.FONT_HERSHEY_SIMPLEX

def sign_detection():
    # get video frame
    areas=[]
    lst_frame_src_array = []
    signs_array = []
    sign_detected = False

    ret, frame_src = video.read()
    #  = frame_src = frame
    # cimg = frame_sc
    # hsv = cv2.cvtColor(frame_sc, cv2.COLOR_BGR2HSV)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    if frame_src is not None:
        frame_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2RGB)
        size = frame_src.shape
        if size is not None:
            detected_instance, predicted_signs = composer.detect_and_classify(frame_src) 

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
                        new_index = str_index.replace(".", "_", 2)
                        new_index=new_index+".png"
                        if str_index == '5.19.1':

                            x=int(lst_frame_src_array[0][0][0][0])-300
                            y=int(lst_frame_src_array[0][0][0][1])
                            h = 400
                            w = 500
                            cropped_frame = frame_src[y:y+h, x:x+w]

                            print(x,y)
                            if (x > 0) and (y > 0):
                                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)   
                                hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
                                lower_blue = np.array([100,150,0])
                                upper_blue = np.array([140,255,255])
                                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                                cropped_frame = cv2.bitwise_and(cropped_frame,cropped_frame, mask= mask)

                                if np.any(mask):
                                    putText_w=int(video_w/6)
                                    putText_h=int(video_h/4)
                                    frame_src = cv2.putText(frame_src, 'PERSON DETECTED',(putText_w,putText_h), font, 2 , (255,0,0), 4, cv2.LINE_AA)
                                else:
                                    pass

                            else:
                                pass
                
            frame_src = cv2.cvtColor(frame_src, cv2.COLOR_RGB2BGR)
            # frame_src=cv2.resize(frame_src,(640,480))
            cv2.imshow('SIGNS', frame_src)
        else:
            pass     
    else:
        pass

def detect_all():
    while (video.isOpened()):
        imgsz = opt.img_size 
        out = opt.output
        source = '0'
        weights = '/home/blackwidow/catkin_ws/src/traffic-light-yolov3-pkg/model/weights/best_model_12.pt'
        half = opt.half
        view_img = opt.view_img
        webcam = source == '0' 

        # Initialize
        device = torch.device('cpu')
        os.makedirs(out, exist_ok=True)  # make new output folder

        # Initialize model
        model = Darknet(opt.cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu' 
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True # default True
            torch.backends.cudnn.benchmark = True 
            dataset = LoadStreams(source, img_size=imgsz, video_src=video)
        else:
            view_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = load_classes(opt.names)
        colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0 , 118, 255)]

        # Run inference
        last_time = time.time()
        fps_counter = 0

        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
        # for path, img, im0s, vid_cap, frame, nframes in dataset:
        for [path, img, im0s, vid_cap] in dataset:
            sign_detection()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = torch_utils.time_synchronized()

            if half:
                pred = pred.float()

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                    multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred): 
                if webcam:  
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:] 
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:
                        if view_img:  # Add bbox to image
                            # label = '%s %.2f' % (names[int(cls)], conf)
                            label = '%s' % (names[int(cls)])
                            print(int(cls))
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                fps_counter+=1
                if time.time() - last_time > 1:
                    last_time = time.time()
                    fps_counter=0

                cv2.imshow('TRAFFIC LIGHT', im0)
                
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-6cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/traffic_light.names', help='*.names path')
    parser.add_argument('--weights', type=str, required=False, help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='outputs', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect_all()
        
        
