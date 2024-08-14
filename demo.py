import argparse
import torch
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np
import math
import torch.nn.functional as F


def run(
    weights_detect='weights/best.pt',
    source='test_image/demo.jpg',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='cpu',
    classes=None,
    agnostic_nms=False,
    line_thickness=3
):
    # Initialize
    device = select_device(device)
    model_detect = DetectMultiBackend(weights_detect, device=device)
    stride, names, pt = model_detect.stride, model_detect.names, model_detect.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Load image
    img_source = cv2.imread(source)
    shape_img_source = img_source.shape
    img_source_array = letterbox(img_source, imgsz, stride=stride, auto=pt)[0]
    shape_img = img_source_array.shape
    img = img_source_array.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = torch.from_numpy(img.copy()).to(device)
    img = img.half() if model_detect.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]


    pred = model_detect(img, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(1)
    

if __name__ == '__main__':
    run()