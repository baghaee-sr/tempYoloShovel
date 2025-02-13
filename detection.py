from config import DRAW_BOXES, IOU_THRESHOLD, CONF_THRESHOLD, TEETH_THRESHOLD, \
    TEETH_AREA_THRESHOLD  # اضافه کردن مقدار جدید
import torch
from coreYoloV5.utils.general import non_max_suppression, scale_boxes
from coreYoloV5.utils.plots import Annotator


def detect_objects(pred, img_shape, im0, names):
    """
    پردازش و شناسایی اشیاء شامل bucket و teeth
    """
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)
    annotator = Annotator(im0, line_width=2)

    bucket_detected = False
    bucket_size = 0
    object_counts = {}

    h_net, w_net = img_shape  # ابعاد ورودی مدل (320x320)
    h_orig, w_orig = im0.shape[:2]  # ابعاد تصویر اصلی

    width_ratio = w_orig / w_net
    height_ratio = h_orig / h_net

    for det in pred:
        if det is not None and len(det):
            det[:, [0, 2]] *= width_ratio  # تبدیل X
            det[:, [1, 3]] *= height_ratio  # تبدیل Y
            det[:, :4] = det[:, :4].round()

            for *xyxy, conf, cls in reversed(det):
                if conf >= CONF_THRESHOLD:
                    class_name = names[int(cls)]

                    # **محاسبه‌ی مساحت کادر و فیلتر کردن ناخن‌های کوچک**
                    box_width = xyxy[2] - xyxy[0]
                    box_height = xyxy[3] - xyxy[1]
                    box_area = box_width * box_height

                    if class_name == "teeth" and box_area < TEETH_AREA_THRESHOLD:
                        continue  # 👈 اگر مساحت ناخن کمتر از حد تعیین‌شده است، نادیده گرفته شود

                    object_counts[class_name] = object_counts.get(class_name, 0) + 1

                    color = (0, 255, 0) if class_name == "bucket" else (255, 0, 0)
                    label = f"{class_name} {conf:.2f}"

                    if DRAW_BOXES:
                        annotator.box_label(xyxy, label, color=color)

                    if class_name == "bucket":
                        bucket_size = box_area
                        bucket_detected = True

    return bucket_detected, bucket_size, object_counts, im0, annotator
