# core/camera_handler.py
import cv2
import threading
from core.yolo_processor import YoloProcessor
from config import config
from core.utils import is_day
import time
import os



class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(config["input_source"])
        self.yolo = YoloProcessor()

        self.frame = None
        self.processed_frame = None
        self.dets = []

        self.lock = threading.Lock()
        self.running = True

        # شروع نخ‌ها
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.capture_thread.start()
        self.processing_thread.start()

        self.last_capture_time = 0

    def capture_frames(self):
        """خواندن فریم‌ها به صورت پشت سر هم"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.frame = frame
            time.sleep(0.01)  # کاهش مصرف CPU

    def process_frames(self):
        """پردازش فریم‌ها با YOLO در یک نخ جدا"""
        while self.running:
            frame = None
            with self.lock:
                if self.frame is not None:
                    frame = self.frame.copy()
            if frame is not None:
                is_daytime = is_day(frame)
                if (is_daytime and config["enable_day"]) or (not is_daytime and config["enable_night"]):
                    processed_frame, dets = self.yolo.process_with_dets(frame)
                else:
                    processed_frame, dets = frame, []

                with self.lock:
                    self.processed_frame = processed_frame
                    self.dets = dets
                    # ذخیره عکس اگر لازم باشد
            self.save_capture_if_needed(frame, 60)
            time.sleep(0.01)

    def get_processed_frame_with_dets(self):
        """دریافت آخرین فریم پردازش‌شده و نتایج YOLO"""
        with self.lock:
            return self.processed_frame, self.dets

    def save_capture_if_needed(self, frame, motion_change):
        if not config.get("capture_enabled", False):
            return
        if motion_change > config.get("motion_threshold", 20):
            now = time.time()
            interval = config.get("capture_interval_sec", 10)
            if now - self.last_capture_time >= interval:
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                log_dir = os.path.join("logs", today, "captures")
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%H-%M-%S")
                # تغییر اندازه
                img_w = config.get("capture_image_width", 640)
                img_h = config.get("capture_image_height", 480)
                frame_resized = cv2.resize(frame, (img_w, img_h))
                filename = os.path.join(log_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame_resized)
                self.last_capture_time = now

    def __del__(self):
        """آزادسازی منابع"""
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()
