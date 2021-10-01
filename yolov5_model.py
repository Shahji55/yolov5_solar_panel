"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(path):
    device = select_device('0')
    weights = path
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride = int(model.stride.max())  # model stride
    
    return model, device

def inference(image, model, device, conf=0.25):
    outputDict = {0:[], 1:[], 2:[]}

    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    print(names)
    imgsz = 1024
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    half = False and device.type != 'cpu'  # half precision only supported on CUDA

    img = letterbox(image, new_shape = imgsz)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
     
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # print(img)
    print(img.shape)
    pred = model(img, augment = False)[0]
    # dataset = LoadImages(image, img_size=imgsz, stride=stride)
    # Apply NMS
    pred = non_max_suppression(pred, conf, 0.45, classes = [0,1,2], agnostic = True)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Process detections
    for det in pred:  # detections per image
        # Append results
        for *xyxy, conf, cls in reversed(det):
            x=xyxy
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

            c = (c1[0] + int((c1[0] + c2[0]) /2) , c1[1] + int((c1[1] + c2[1]) /2))
            dictCordinates= {'x': c[0], 'y':c[1]}
            outputDict[int(cls)].append(dictCordinates)

    return outputDict