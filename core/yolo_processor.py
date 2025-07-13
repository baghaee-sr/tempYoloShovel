# core/yolo_processor.py
import cv2
import torch
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.utils.general import non_max_suppression
from coreYoloV5.utils.plots import Annotator, colors
from coreYoloV5.models.common import DetectMultiBackend
from config import config

def boxes_intersect(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = xB - xA
    interHeight = yB - yA
    return interWidth > 0 and interHeight > 0

class YoloProcessor:
    def __init__(self):
        self.config = config
        self.device = select_device("")
        self.model = DetectMultiBackend(config["model_path"], device=self.device)
        self.names = self.model.names
        self.stride = int(self.model.stride)

    def process_with_dets(self, im0):
        h0, w0 = im0.shape[:2]
        input_w, input_h = 640, 640
        im_resized = cv2.resize(im0, (input_w, input_h))

        width_ratio = w0 / input_w
        height_ratio = h0 / input_h

        img = torch.from_numpy(im_resized).to(self.device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)

        annotator = Annotator(im0, line_width=2)
        dets = []

        # ذخیره باکس bucket ها و teeth ها جداگانه برای بررسی همپوشانی
        bucket_boxes = []
        teeth_candidates = []

        for det in pred:
            if det is not None and len(det):
                det[:, [0, 2]] *= width_ratio
                det[:, [1, 3]] *= height_ratio
                det[:, :4] = det[:, :4].round()
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    if label == "bucket":
                        bucket_boxes.append({"bbox": [x1, y1, x2, y2], "area": area})
                    elif label == "teeth":
                        teeth_candidates.append({"bbox": [x1, y1, x2, y2], "area": area})

                    if self.config.get("draw_boxes", True):
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # حالا فقط teeth هایی که حداقل با یک bucket همپوشانی دارند انتخاب کن
        teeth_filtered = []
        for tooth in teeth_candidates:
            for bucket in bucket_boxes:
                if boxes_intersect(tooth["bbox"], bucket["bbox"]):
                    teeth_filtered.append(tooth)
                    break  # یکبار یافتن همپوشانی کافی است

        # ساخت لیست نهایی دت‌ها با bucket ها و teeth های فیلتر شده
        for bucket in bucket_boxes:
            dets.append({"class": "bucket", "area": bucket["area"], "bbox": bucket["bbox"]})
        for tooth in teeth_filtered:
            dets.append({"class": "teeth", "area": tooth["area"], "bbox": tooth["bbox"]})

        # --- رسم متن کالیبراسیون روی تصویر ---
        if self.config.get("show_area_values", False) and len(dets) > 0:
            lines = []
            bucket_idx = 1
            teeth_idx = 1
            for det in dets:
                label = det["class"]
                area = int(det["area"])
                if label == "bucket":
                    lines.append(f"Bucket #{bucket_idx}: {area}")
                    bucket_idx += 1
                elif label == "teeth":
                    lines.append(f"Teeth #{teeth_idx}: {area}")
                    teeth_idx += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            color_text = (0, 255, 0)  # سبز روشن

            margin = 10
            line_height = 20
            x = margin
            y = im0.shape[0] - margin

            box_width = 250
            box_height = line_height * len(lines) + 10

            cv2.rectangle(im0, (x - 5, y - box_height), (x + box_width, y + 5), (0, 0, 0), cv2.FILLED)

            for i, line in enumerate(lines):
                text_pos = (x, y - line_height * (len(lines) - i - 1))
                cv2.putText(im0, line, text_pos, font, font_scale, color_text, thickness, cv2.LINE_AA)

        return annotator.result(), dets
