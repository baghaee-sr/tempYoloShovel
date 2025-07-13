import sys
import cv2
import torch
import numpy as np
import threading  # 📌 اضافه کردن Threading برای پردازش موازی
from PySide6.QtWidgets import QApplication, QMainWindow, QStatusBar, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.utils.general import non_max_suppression
from coreYoloV5.utils.plots import Annotator, colors
from coreYoloV5.models.common import DetectMultiBackend

import time  # 📌 اضافه کردن برای محاسبه FPS

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()
        self.loadModel()
        self.running = True
        self.frame = None
        self.processed_frame = None

        # ✅ متغیرهای FPS
        self.frame_count = 0
        self.last_time = time.time()

        # اجرای پردازش YOLO در Thread جداگانه
        self.processing_thread = threading.Thread(target=self.yolo_processing, daemon=True)
        self.processing_thread.start()


    def initUI(self):
        """ تنظیم رابط کاربری """
        self.showFullScreen()
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.left_label = QLabel("سلامت سیستم", self)
        self.left_label.setAlignment(Qt.AlignLeft)
        self.left_label.setStyleSheet("background-color: blue; color: white; padding: 5px;")
        self.right_label = QLabel("سلامت ناخن", self)
        self.right_label.setAlignment(Qt.AlignRight)
        self.right_label.setStyleSheet("background-color: gray; color: white; padding: 5px;")

        self.status_bar.addPermanentWidget(self.left_label, 1)
        self.status_bar.addPermanentWidget(self.right_label, 1)

    def loadModel(self):
        """ بارگذاری مدل YOLOv5 """
        self.device = select_device("")  # انتخاب CPU یا GPU
        self.model = DetectMultiBackend("weights/img1024_notSorting.pt", device=self.device)
        self.names = self.model.names  # کلاس‌های مدل
        self.stride = int(self.model.stride)  # مقدار stride مدل

    def initCamera(self):
        """ بارگذاری ویدیو یا دوربین """
        self.cap = cv2.VideoCapture('files/input_video.mp4')
        # self.cap = cv2.VideoCapture('output_2025-01-31_11-48-58.mp4')

        # تنظیم تایمر برای نمایش فریم‌ها
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # نمایش هر ۳۰ میلی‌ثانیه

    def yolo_processing(self):
        """ پردازش YOLO در Thread جداگانه برای افزایش سرعت """
        while self.running:
            if self.frame is not None:
                self.processed_frame = self.process_frame(self.frame.copy())  # پردازش YOLO

    def update_frame(self):
        """ دریافت فریم جدید، پردازش و نمایش FPS """
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame  # ذخیره فریم برای پردازش
            if self.processed_frame is not None:
                display_frame = self.processed_frame
            else:
                display_frame = frame

            # ✅ محاسبه FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_time

            if elapsed_time >= 1.0:  # هر 1 ثانیه یک بار بروزرسانی شود
                fps = self.frame_count / elapsed_time
                self.left_label.setText(f"FPS: {fps:.2f}")  # نمایش FPS در نوار وضعیت
                self.frame_count = 0
                self.last_time = current_time

            # تبدیل فریم به RGB برای نمایش در PySide6
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            q_img = QImage(display_frame.data, w, h, ch * w, QImage.Format_RGB888)

            # تغییر اندازه تصویر برای نمایش مناسب در QLabel
            pixmap = QPixmap.fromImage(q_img)
            screen_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(screen_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(scaled_pixmap)  # نمایش در QLabel

    def process_frame(self, im0):
        """ YOLO روی تصویر کوچک‌تر اجرا شده و خروجی به اندازه اصلی بازگردانده می‌شود """

        # ابعاد اصلی تصویر
        h0, w0 = im0.shape[:2]

        # 🔹 تغییر اندازه تصویر فقط برای پردازش YOLO (مثلاً 640x480)
        input_w, input_h = 640, 640
        im_resized = cv2.resize(im0, (input_w, input_h))

        # نسبت تغییر اندازه
        width_ratio = w0 / input_w
        height_ratio = h0 / input_h

        # تبدیل به فرمت YOLO: (C, H, W)
        img = torch.from_numpy(im_resized).to(self.device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # پردازش با مدل YOLO
        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)

        # رسم خروجی روی تصویر اصلی
        annotator = Annotator(im0, line_width=2)
        for det in pred:
            if det is not None and len(det):
                det[:, [0, 2]] *= width_ratio
                det[:, [1, 3]] *= height_ratio
                det[:, :4] = det[:, :4].round()

                for *xyxy, conf, cls in reversed(det):
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        return annotator.result()  # نمایش در GUI با اندازه اصلی

    def closeEvent(self, event):
        """ آزاد کردن منابع هنگام بستن برنامه """
        self.running = False  # متوقف کردن پردازش
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
