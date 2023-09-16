#!/home/katya/agv_ws/src/traffic-light-yolov3-pkg/.venv/bin/python 
import argparse
import rospy
import rospkg
from std_msgs.msg import Int8


from model.scripts.models import *  
from utils.datasets import *
from utils.utils import *


def detect():
    ##ROS SECTION##
    pub = rospy.Publisher('detectTrafficLights', Int8, queue_size=10)
    result = 0
    rospy.init_node('TrafficLightsDetector', anonymous=True)
    rospack = rospkg.RosPack()
    # rate = rospy.Rate(10) # 10hz
    PATH_TO_WEIGHTS=f"{rospack.get_path('traffic-light-yolov3-pkg')}/model/weights/best_model_12.pt"
    ##ROS SECTION END##

    imgsz = opt.img_size 
    out = opt.output
    source = '0'
    weights = PATH_TO_WEIGHTS
    half = opt.half
    view_img = opt.view_img
    webcam = source == '0' 

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    if webcam:
        view_img = True # default True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0 , 118, 255)]

    # Run inference
    last_time = time.time()
    fps_counter = 0

    # t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    # for [path, img, im0s, vid_cap, frame, nframes] in dataset:
    for [path, img, im0s, vid_cap] in dataset:
        # view_img = True
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
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

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    if view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (names[int(cls)])
                        result=int(cls)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        rospy.loginfo(result)
                        pub.publish(result)

            fps_counter+=1
            if time.time() - last_time > 1:
                last_time = time.time()
                fps_counter=0

            # Stream results
            if view_img == True:
                cv2.imshow('window', im0)
                # print(type(im0),im0)
            if rospy.is_shutdown():
                cv2.destroyAllWindows()
                raise StopIteration

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
    parser.add_argument('--view-img', action='store_true', default = 'true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()