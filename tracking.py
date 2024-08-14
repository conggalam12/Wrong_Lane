from ocsort.ocsort import OCSort
import cv2
import torch
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from models.common import DetectMultiBackend
import numpy as np
from polygon import draw_polygon , read_file_config , determine_side
def relu(x):
    return max(0, int(x))
def create_model(weights_detect='weights/best.pt', device='cpu'):
    device = select_device(device)
    model_detect = DetectMultiBackend(weights_detect, device=device)
    return model_detect
def pre_process(img, imgsz, stride, pt):
    img_source_array = letterbox(img, imgsz, stride=stride, auto=pt)[0]
    img = np.ascontiguousarray(img_source_array[:, :, ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
            # expand dims
        img = img.unsqueeze(0)
    return img

tracker = OCSort(det_thresh=0.45, iou_threshold=0.3, use_byte=False)

video = cv2.VideoCapture("/home/congnt/congnt/ATIN/test/video_test/demo.mp4")

model = create_model(weights_detect='/home/congnt/congnt/ATIN/wrong_lane/yolov5/weights/anh_hoai.pt')
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640,640), s=stride)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
polygon = read_file_config("../config/data.json")
data = {}
while True:
    ret, frame = video.read()

    frame_origin = frame.copy()
    id_dict = {}
    
    frame_processed = pre_process(frame, imgsz, stride, pt)
    pred = model(frame_processed)  
    # pred = non_max_suppression(pred,conf_thres=0.25,classes=[2,3,5,7], max_det=1000)
    pred = non_max_suppression(pred,conf_thres=0.25,classes=None, max_det=1000)
    det = pred[0]
    det[:, :4] = scale_boxes(frame_processed.shape[2:], det[:, :4], frame_origin.shape).round()
    outputs = tracker.update(det.cpu(), frame_origin)
    
    if len(outputs):
        for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
            x1, y1, x2, y2 = list(map(relu, output[0:4]))
            ids = int(output[4])
            cls = int(output[5])

            id_dict[ids] = [x1, y1, x2, y2, cls]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            center = (cx, cy)
            color = (0, 255, 0)  
            
            tracking = determine_side(center,polygon)
            if tracking == "inside":
                data[ids] = tracking
            if tracking == "straight" and data.get(ids):
                data[ids] = tracking
            if tracking == "left" and data.get(ids):
                label = f"ID: {ids} " + "Error"
                color = (255, 0, 0)  
            label = f"ID: {ids} " + tracking
            # else:
            #     label = f"ID: {ids} "
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    draw_polygon(frame,polygon)
    cv2.imshow("Vehicle Tracking", frame)

    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()