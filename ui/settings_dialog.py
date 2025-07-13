# ui/settings_dialog.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QCheckBox, QPushButton,
    QSpinBox, QLabel, QHBoxLayout
)
from config import config

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("تنظیمات")
        self.layout = QVBoxLayout(self)

        # تنظیمات پردازش روز و شب
        self.day_check = QCheckBox("پردازش در روز")
        self.night_check = QCheckBox("پردازش در شب")
        self.box_check = QCheckBox("کشیدن باکس")
        self.show_area_check = QCheckBox("نمایش و لاگ مقدار مساحت‌ها (کالیبراسیون)")

        self.day_check.setChecked(config.get("enable_day", True))
        self.night_check.setChecked(config.get("enable_night", False))
        self.box_check.setChecked(config.get("draw_boxes", True))
        self.show_area_check.setChecked(config.get("show_area_values", False))

        self.layout.addWidget(self.day_check)
        self.layout.addWidget(self.night_check)
        self.layout.addWidget(self.box_check)
        self.layout.addWidget(self.show_area_check)

        # --- تنظیمات ذخیره خودکار عکس ---
        self.capture_check = QCheckBox("ذخیره خودکار عکس هنگام کار کردن شاول")
        self.capture_check.setChecked(config.get("capture_enabled", False))
        self.layout.addWidget(self.capture_check)

        interval_layout = QHBoxLayout()
        interval_label = QLabel("بازه ذخیره (ثانیه):")
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(600)
        self.interval_spin.setValue(config.get("capture_interval_sec", 10))
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spin)
        self.layout.addLayout(interval_layout)

        size_layout = QHBoxLayout()
        width_label = QLabel("عرض عکس:")
        self.width_spin = QSpinBox()
        self.width_spin.setMinimum(64)
        self.width_spin.setMaximum(1920)
        self.width_spin.setValue(config.get("capture_image_width", 640))
        height_label = QLabel("ارتفاع عکس:")
        self.height_spin = QSpinBox()
        self.height_spin.setMinimum(64)
        self.height_spin.setMaximum(1080)
        self.height_spin.setValue(config.get("capture_image_height", 480))
        size_layout.addWidget(width_label)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(height_label)
        size_layout.addWidget(self.height_spin)
        self.layout.addLayout(size_layout)

        # دکمه ذخیره
        self.save_btn = QPushButton("ذخیره")
        self.save_btn.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_btn)

    def save_settings(self):
        config["enable_day"] = self.day_check.isChecked()
        config["enable_night"] = self.night_check.isChecked()
        config["draw_boxes"] = self.box_check.isChecked()
        config["show_area_values"] = self.show_area_check.isChecked()
        config["capture_enabled"] = self.capture_check.isChecked()
        config["capture_interval_sec"] = self.interval_spin.value()
        config["capture_image_width"] = self.width_spin.value()
        config["capture_image_height"] = self.height_spin.value()
        self.accept()
