# ui_settings.py
import json
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QPushButton
from config import CONF_THRESHOLD, IOU_THRESHOLD, TEETH_THRESHOLD

CONFIG_PATH = "config.json"  # مسیر ذخیره تنظیمات موقتی


class SettingsPage(QWidget):
    """ صفحه تنظیمات برای تغییر مقادیر CONF_THRESHOLD، IOU_THRESHOLD و TEETH_THRESHOLD در حین اجرا """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚙ تنظیمات")
        self.setGeometry(250, 250, 400, 300)

        layout = QVBoxLayout()

        # تنظیم مقدار CONF_THRESHOLD
        self.conf_label = QLabel("حد آستانه اطمینان (Confidence Threshold):")
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setRange(10, 100)
        self.conf_spinbox.setValue(int(CONF_THRESHOLD * 100))

        # تنظیم مقدار IOU_THRESHOLD
        self.iou_label = QLabel("حد آستانه IOU (Non-Max Suppression):")
        self.iou_spinbox = QSpinBox()
        self.iou_spinbox.setRange(10, 100)
        self.iou_spinbox.setValue(int(IOU_THRESHOLD * 100))

        # تنظیم مقدار TEETH_THRESHOLD
        self.teeth_label = QLabel("حداقل تعداد Teeth:")
        self.teeth_spinbox = QSpinBox()
        self.teeth_spinbox.setRange(1, 10)
        self.teeth_spinbox.setValue(TEETH_THRESHOLD)

        # دکمه ذخیره تنظیمات
        self.save_button = QPushButton("💾 ذخیره تغییرات")
        self.save_button.clicked.connect(self.save_settings)

        # اضافه کردن ویجت‌ها به لایه
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_spinbox)
        layout.addWidget(self.iou_label)
        layout.addWidget(self.iou_spinbox)
        layout.addWidget(self.teeth_label)
        layout.addWidget(self.teeth_spinbox)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_settings(self):
        """ ذخیره تنظیمات در یک فایل موقت JSON """
        new_settings = {
            "CONF_THRESHOLD": self.conf_spinbox.value() / 100,
            "IOU_THRESHOLD": self.iou_spinbox.value() / 100,
            "TEETH_THRESHOLD": self.teeth_spinbox.value()
        }

        with open(CONFIG_PATH, "w") as file:
            json.dump(new_settings, file, indent=4)

        self.close()  # بستن صفحه تنظیمات
