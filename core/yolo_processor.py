# core/yolo_processor.py
import torch
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.utils.general import non_max_suppression
from coreYoloV5.utils.plots import Annotator, colors
from coreYoloV5.models.common import DetectMultiBackend
from config import config
import cv2
import os
from datetime import datetime

class YoloProcessor:
    def __init__(self):
        self.config = config
        self.device = select_device("")
        self.model = DetectMultiBackend(config["model_path"], device=self.device)
        self.names = self.model.names
        self.stride = int(self.model.stride)

    def process(self, im0):
        # تغییر اندازه برای YOLO
        input_w, input_h = 640, 640
        im_resized = cv2.resize(im0, (input_w, input_h))
        width_ratio = im0.shape[1] / input_w
        height_ratio = im0.shape[0] / input_h

        img = torch.from_numpy(im_resized).to(self.device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)

        if config["draw_boxes"]:
            annotator = Annotator(im0, line_width=2)
            for det in pred:
                if det is not None and len(det):
                    det[:, [0, 2]] *= width_ratio
                    det[:, [1, 3]] *= height_ratio
                    det[:, :4] = det[:, :4].round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
            return annotator.result()
        else:
            return im0

    def process_with_dets(self, im0):
        # ابعاد اصلی تصویر
        h0, w0 = im0.shape[:2]
        input_w, input_h = 640, 640
        im_resized = cv2.resize(im0, (input_w, input_h))

        width_ratio = w0 / input_w
        height_ratio = h0 / input_h

        img = torch.from_numpy(im_resized).to(self.device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # --- این خط بسیار مهم است:
        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)

        annotator = Annotator(im0, line_width=2)
        dets = []
        show_areas = self.config.get("show_area_values", False)
        for det in pred:
            if det is not None and len(det):
                det[:, [0, 2]] *= width_ratio
                det[:, [1, 3]] *= height_ratio
                det[:, :4] = det[:, :4].round()
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    dets.append({"class": label, "area": area, "bbox": [x1, y1, x2, y2], "conf": float(conf)})
                    # ← نمایش مقدار مساحت روی تصویر
                    if show_areas and config["draw_boxes"]:
                        area_label = f"{label} area: {area}"
                        annotator.text((x1, y1 - 10), area_label, txt_color=(255, 0, 0))

                    if self.config.get("draw_boxes", True):
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
        return annotator.result(), dets

