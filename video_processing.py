import time
import cv2
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.models.common import DetectMultiBackend
from detection import detect_objects
from config import (
    WEIGHTS_PATH, VIDEO_PATH, BUCKET_THRESHOLD, USE_CAMERA,
    STREAM_URL, TEETH_THRESHOLD, IOU_THRESHOLD, CONF_THRESHOLD, IMG_SIZE, MIN_DUMP_TIME, DRAW_BOXES
)


class VideoProcessingThread(QThread):
    update_processed_frame = pyqtSignal(QImage)
    update_status = pyqtSignal(str, str)
    update_object_counts = pyqtSignal(dict)
    update_fps = pyqtSignal(float)

    def __init__(self, weights=WEIGHTS_PATH):
        super().__init__()
        self.weights = weights
        self.running = True
        self.in_dumping_mode = False
        self.max_teeth_count = 0
        self.dump_start_time = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # انتخاب منبع ویدیو (دوربین یا فایل)
        self.source = STREAM_URL if USE_CAMERA else VIDEO_PATH
        self.capture = cv2.VideoCapture(self.source)

        if USE_CAMERA:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def run(self):
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device)
        names = model.names

        while self.running:
            start_time = time.time()
            ret, frame = self.capture.read()
            if not ret:
                self.update_status.emit("❌ خطا در دریافت فریم", "red")
                continue

            im0 = frame.copy()  # حفظ تصویر اصلی برای ترسیم Bounding Box

            # تغییر اندازه‌ی تصویر به رزولوشن YOLOv5
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_resized = img_resized[:, :, ::-1].copy()  # تبدیل BGR به RGB
            img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            # اجرای YOLOv5 روی تصویر
            pred = model(img_tensor)

            # پردازش خروجی مدل
            bucket_detected, bucket_size, object_counts, im0_processed, annotator = detect_objects(
                pred, img_tensor.shape[2:], im0, names
            )

            # محاسبه FPS و ارسال مقدار
            fps = 1 / (time.time() - start_time)
            self.update_fps.emit(fps)

            # دریافت تعداد teeth
            current_teeth_count = object_counts.get("teeth", 0)

            # بررسی ورود به **حالت تخلیه**
            if bucket_detected and bucket_size > BUCKET_THRESHOLD:
                if not self.in_dumping_mode:
                    self.in_dumping_mode = True
                    self.dump_start_time = time.time()
                    self.max_teeth_count = 0

                self.max_teeth_count = max(self.max_teeth_count, current_teeth_count)

            else:
                if self.in_dumping_mode:
                    dump_duration = time.time() - self.dump_start_time

                    if dump_duration < MIN_DUMP_TIME:
                        continue  # زمان تخلیه کافی نیست، پس در تخلیه بمان

                    self.in_dumping_mode = False  # خروج از تخلیه

                    if self.max_teeth_count < TEETH_THRESHOLD:
                        status_text = f"⚠️ اخطار: تعداد ناخن ناکافی ({self.max_teeth_count}/{TEETH_THRESHOLD})"
                        status_color = "red"
                    else:
                        status_text = "✅ سلامت سیستم"
                        status_color = "green"

                    self.update_status.emit(status_text, status_color)

            # ✅ اینجا منطق صحیح نمایش فقط teeth:
            force_teeth_only = (
                not DRAW_BOXES
                and self.in_dumping_mode
                and self.max_teeth_count == TEETH_THRESHOLD
            )

            # پردازش نهایی برای رسم BBox با توجه به شرایط
            bucket_detected, bucket_size, object_counts, im0_processed, annotator = detect_objects(
                pred, img_tensor.shape[2:], im0, names, force_teeth_only=force_teeth_only
            )

            object_counts["bucketStatus"] = "bucketStatus = True" if bucket_detected else "bucketStatus = False"
            self.update_object_counts.emit(object_counts)

            im0_processed = annotator.result()
            rgb_processed = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_processed.shape
            bytes_per_line = ch * w
            qt_processed = QImage(rgb_processed.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_processed_frame.emit(qt_processed)

    def stop(self):
        self.running = False
        self.capture.release()
        self.quit()
