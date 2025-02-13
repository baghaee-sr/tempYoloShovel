import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from video_processing import VideoProcessingThread
from config import WEIGHTS_PATH

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shovel")
        self.showFullScreen()

        # ایجاد ویجت اصلی
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # لیبل نمایش تصویر پردازش‌شده
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setStyleSheet("background-color: black;")

        # جدول وضعیت (2 ستون: فقط وضعیت سیستم و نمایش تعداد teeth در زمان تخلیه)
        self.status_table = QTableWidget(1, 2)  # حذف ستون "bucketStatus"
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.horizontalHeader().setVisible(False)
        self.status_table.setShowGrid(False)
        self.status_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)

        # استایل جدول
        self.status_table.setStyleSheet("""
            QTableWidget {
                font-size: 20px;
                font-weight: bold;
                background-color: #2E2E2E;
                color: white;
                border: none;
            }
            QTableWidget::item {
                text-align: center;
            }
        """)

        self.status_table.setFixedHeight(40)
        self.status_table.setColumnWidth(0, self.width() // 2)
        self.status_table.setColumnWidth(1, self.width() // 2)

        # مقداردهی اولیه جدول
        self.status_table.setItem(0, 0, self.create_table_item("✅ سلامت سیستم"))
        self.status_table.setItem(0, 1, self.create_table_item(""))

        # افزودن ویجت‌ها به صفحه
        layout.addWidget(self.processed_label, 1)
        layout.addWidget(self.status_table, 0)  # فقط نمایش وضعیت سیستم و تعداد teeth

        # اجرای پردازش ویدیو
        self.thread = VideoProcessingThread(weights=WEIGHTS_PATH)
        self.thread.update_processed_frame.connect(self.update_processed_frame)
        self.thread.update_status.connect(self.update_status)
        self.thread.update_object_counts.connect(self.update_object_counts)

        self.thread.start()

    def create_table_item(self, text):
        """ ایجاد آیتم جدول با استایل مناسب """
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def update_processed_frame(self, frame):
        """ بروزرسانی تصویر پردازش شده """
        pixmap = QPixmap.fromImage(frame)
        self.processed_label.setPixmap(pixmap)
        self.processed_label.setScaledContents(True)

    def update_status(self, status, color):
        """ بروزرسانی وضعیت سیستم در جدول """
        item = self.create_table_item(status)
        item.setForeground(Qt.green if color == "green" else Qt.red)
        self.status_table.setItem(0, 0, item)

    def update_object_counts(self, object_counts):
        """ در زمان تخلیه، فقط بیشترین تعداد teeth را نمایش بده """
        if "bucketStatus" in object_counts:
            object_counts.pop("bucketStatus")  # حذف وضعیت bucket

        if self.thread.in_dumping_mode:  # فقط در زمان تخلیه نمایش بده
            max_teeth = self.thread.max_teeth_count
            self.status_table.setItem(0, 1, self.create_table_item(f"🔍  شناسایی  تعداد ناخن: {max_teeth}"))
        else:
            self.status_table.setItem(0, 1, self.create_table_item(""))  # بعد از تخلیه مقدار را پاک کن

    def closeEvent(self, event):
        """ متوقف کردن پردازش هنگام بستن برنامه """
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
