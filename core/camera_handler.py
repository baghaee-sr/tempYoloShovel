import cv2
import os
import time
import threading
from datetime import datetime
from core.yolo_processor import YoloProcessor
from config import config
from core.utils import is_day

class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(config["input_source"])
        self.yolo = YoloProcessor()
        self.latest_frame = None         # آخرین فریم دریافتی
        self.latest_result = (None, [])  # (processed_frame, dets)
        self.latest_motion = 0
        self.last_capture_time = 0       # زمان آخرین ذخیره عکس
        self.running = True
        self.lock = threading.Lock()     # برای thread-safe بودن

        # ---- نخ جداگانه برای Capture و ذخیره عکس
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.capture_thread.start()
        self.processing_thread.start()

    def capture_frames(self):
        prev_gray = None

        # خواندن FPS ویدیو (اگر نداشت ۳۰ فرض کن)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:  # اگر ویدیوی تو weird بود
            fps = 30
        frame_duration = 1.0 / fps

        while self.running:
            start_time = time.time()

            ret, frame = self.cap.read()
            if ret:
                # محاسبه motion و ... (کدهای خودت)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion_change = 0
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_change = diff.mean()
                prev_gray = gray

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_motion = motion_change

                self.save_capture_if_needed(frame, motion_change)
            else:
                time.sleep(0.02)

            # توقف برای هماهنگ شدن با سرعت ویدیوی واقعی
            elapsed = time.time() - start_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def process_frames(self):
        while self.running:
            # فقط آخرین فریم را بردار (thread-safe)
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                motion_change = self.latest_motion

            if frame is None:
                time.sleep(0.02)
                continue

            # تشخیص روز/شب
            is_daytime = is_day(frame)
            process_flag = (
                (is_daytime and config["enable_day"]) or
                (not is_daytime and config["enable_night"])
            )

            if process_flag:
                processed_frame, dets = self.yolo.process_with_dets(frame)
            else:
                processed_frame, dets = frame, []

            # نتیجه پردازش را برای UI ذخیره کن
            self.latest_result = (processed_frame, dets)

            time.sleep(config.get("fps", 30)/1000.0)

    def get_processed_frame_with_dets(self):
        return self.latest_result

    def get_last_motion(self):
        with self.lock:
            return self.latest_motion

    def save_capture_if_needed(self, frame, motion_change):
        if not config.get("capture_enabled", False):
            return
        if motion_change > config.get("motion_threshold", 20):
            now = time.time()
            interval = config.get("capture_interval_sec", 10)
            if now - self.last_capture_time >= interval:
                today = datetime.now().strftime("%Y-%m-%d")
                log_dir = os.path.join("logs", today, "captures")
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%H-%M-%S")
                img_w = config.get("capture_image_width", 640)
                img_h = config.get("capture_image_height", 480)
                frame_resized = cv2.resize(frame, (img_w, img_h))
                filename = os.path.join(log_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame_resized)
                self.last_capture_time = now

    def __del__(self):
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()
