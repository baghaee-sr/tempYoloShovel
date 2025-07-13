from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from ui.settings_dialog import SettingsDialog
from core.camera_handler import CameraHandler
from core.bucket_monitor import BucketMonitor
from config import config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shovel vision")
        self.showFullScreen()

        # ---- لایه اصلی ----
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- نوار عنوان بالا ----
        title_bar = QLabel("شاول ویژن", self)
        title_bar.setAlignment(Qt.AlignCenter)
        title_bar.setStyleSheet("background-color: #757575; color: #fff; font-size: 10px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title_bar)

        # ---- بخش مرکزی (نمایش تصویر/ویدیو) ----
        self.image_label = QLabel(main_widget)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #757575; border: none; font-size: 22px; font-weight: bold;")
        self.image_label.setText("این بخش مرکزی برنامه است و محل نمایش فیلم/تصویر پردازش شده")
        main_layout.addWidget(self.image_label, stretch=1)

        # ---- نوار وضعیت پایین ----
        status_bar_widget = QWidget(main_widget)
        status_bar_layout = QHBoxLayout(status_bar_widget)
        status_bar_layout.setContentsMargins(0, 0, 0, 0)
        status_bar_layout.setSpacing(0)

        # --- بخش ۱: دکمه‌ها (راست) ---
        btns_widget = QWidget()
        btns_layout = QHBoxLayout(btns_widget)
        btns_layout.setContentsMargins(0,0,0,0)
        btns_layout.setSpacing(0)

        self.settings_btn = QPushButton("تنظیمات")
        self.settings_btn.setMinimumWidth(160)
        self.settings_btn.setStyleSheet("background-color: #388e3c; color: #fff; font-size: 18px; font-weight: bold;")
        self.settings_btn.clicked.connect(self.show_settings)

        self.log_btn = QPushButton("نمایش لاگ")
        self.log_btn.setMinimumWidth(160)
        self.log_btn.setStyleSheet("background-color: #388e3c; color: #fff; font-size: 18px; font-weight: bold;")
        # فعلاً بدون رویداد

        btns_layout.addWidget(self.settings_btn)
        btns_layout.addWidget(self.log_btn)
        btns_widget.setMinimumWidth(350)
        status_bar_layout.addWidget(btns_widget)

        # --- بخش ۲: وضعیت سیستم (وسط) ---
        self.status_label = QLabel("سلامت سیستم")
        self.status_label.setAlignment(Qt.AlignVCenter)
        self.status_label.setStyleSheet("color: #ffffff; font-size: 22px; font-weight: bold; background: #333333; padding: 5px 0;")
        self.status_label.setMinimumWidth(350)
        status_bar_layout.addWidget(self.status_label)

        # --- بخش ۳: پیام مانیتورینگ (چپ) ---
        self.event_label = QLabel("")
        self.event_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.event_label.setStyleSheet("color: #ffffff; font-size: 22px; font-weight: bold; background: #333333; padding: 5px 10px;")
        self.event_label.setMinimumWidth(350)
        status_bar_layout.addWidget(self.event_label, stretch=1)


        main_layout.addWidget(status_bar_widget)
        self.setCentralWidget(main_widget)

        # ---- پردازش و مانیتورینگ ----
        self.camera = CameraHandler()
        self.monitor = BucketMonitor(config)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def show_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()
        self.status_label.setText("تنظیمات باز شد!")

    def update_frame(self):
        frame, dets = self.camera.get_processed_frame_with_dets()
        if frame is not None:
            rgb = frame[..., ::-1]
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            screen_size = self.image_label.size()
            if screen_size.width() == 0 or screen_size.height() == 0:
                screen_size = self.size()
            scaled_pixmap = pixmap.scaled(
                screen_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

            # --- وضعیت الگوریتم سلامت ---
            bucket_area = 0
            teeth_areas = []
            for det in dets:
                if det["class"] == "bucket":
                    bucket_area = det["area"]
                elif det["class"] == "teeth":
                    teeth_areas.append(det["area"])
            motion_change = 0
            self.monitor.update(bucket_area, teeth_areas, motion_change)
            current_status = self.monitor.get_status()
            self.status_label.setText(current_status)

            # پیام مانیتورینگ وسط (لحظه‌ای)
            event_msg = self.monitor.get_event()
            if event_msg:
                self.event_label.setText(event_msg)
                # پاک‌سازی پیام بعد از ۲ ثانیه
                QTimer.singleShot(3000, lambda: self.event_label.setText(""))
                # پیام رو بعد از نمایش خالی کن (تا دوباره تکراری نشه)
                self.monitor.clear_event()
