import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import cv2
import torch
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.utils.dataloaders import LoadImages
from coreYoloV5.utils.general import non_max_suppression, scale_boxes
from coreYoloV5.utils.plots import Annotator, colors
from coreYoloV5.models.common import DetectMultiBackend

class VideoProcessingThread(QThread):
    update_original_frame = pyqtSignal(QImage)
    update_processed_frame = pyqtSignal(QImage)
    update_status = pyqtSignal(str, str)

    def __init__(self, weights, source, device=""):
        super().__init__()
        self.weights = weights
        self.source = source
        self.device = device
        self.running = True

    def run(self):
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device)
        stride, names = model.stride, model.names

        dataset = LoadImages(self.source, img_size=(640, 640), stride=stride)
        for path, img, im0s, _, _ in dataset:
            if not self.running:
                break

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)
            pred = non_max_suppression(pred, 0.25, 0.45)

            im0_original = im0s[0].copy() if isinstance(im0s, list) else im0s
            im0_processed = im0_original.copy()
            annotator = Annotator(im0_processed, line_width=2)

            teeth_count = 0

            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0_processed.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f"{names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                        if names[int(cls)] == "teeth":
                            teeth_count += 1

            if teeth_count >= 5:
                status_text = "سلامت ماشین"
                status_color = "green"
            else:
                status_text = "اخطار بررسی سلامت ماشین"
                status_color = "red"

            self.update_status.emit(status_text, status_color)

            im0_processed = annotator.result()

            rgb_original = cv2.cvtColor(im0_original, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_original.shape
            bytes_per_line = ch * w
            qt_original = QImage(rgb_original.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_original_frame.emit(qt_original)

            rgb_processed = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_processed.shape
            bytes_per_line = ch * w
            qt_processed = QImage(rgb_processed.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_processed_frame.emit(qt_processed)

    def stop(self):
        self.running = False
        self.quit()

class MainApp(QMainWindow):
    def __init__(self, weights, source):
        super().__init__()
        self.setWindowTitle("YOLO Video Processor")
        self.setGeometry(0, 0, QApplication.desktop().screenGeometry().width(), QApplication.desktop().screenGeometry().height())
        self.setWindowState(Qt.WindowFullScreen)

        # Stack for different screens
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        # Welcome Screen
        self.welcome_screen = QWidget()
        self.logo_label = QLabel("Welcome to YOLO Processor", self.welcome_screen)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("font-size: 48px; font-weight: bold;")
        self.start_button = QPushButton("Start", self.welcome_screen)
        self.start_button.setStyleSheet("font-size: 24px;")
        self.start_button.clicked.connect(self.show_main_screen)

        welcome_layout = QVBoxLayout(self.welcome_screen)
        welcome_layout.addWidget(self.logo_label)
        welcome_layout.addWidget(self.start_button)
        self.stack.addWidget(self.welcome_screen)

        # Main Video Processing Screen with styles
        self.main_screen = QWidget()
        self.splitter = QSplitter(self.main_screen)
        self.splitter.setGeometry(0, 0, self.width(), self.height() - 100)

        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)

        self.splitter.addWidget(self.original_label)
        self.splitter.addWidget(self.processed_label)
        self.splitter.setSizes([self.width() // 2, self.width() // 2])

        self.status_label = QLabel(self.main_screen)
        self.status_label.setGeometry(0, self.height() - 100, self.width(), 100)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 32px; font-weight: bold; background-color: lightgray;")

        self.settings_button = QPushButton("Settings", self.main_screen)
        self.settings_button.setGeometry(self.width() - 150, 10, 140, 50)
        self.settings_button.setStyleSheet("font-size: 18px;")
        self.settings_button.clicked.connect(self.show_settings_screen)

        self.stack.addWidget(self.main_screen)

        # Settings Screen
        self.settings_screen = QWidget()
        self.back_button = QPushButton("Back", self.settings_screen)
        self.back_button.setGeometry(10, 10, 140, 50)
        self.back_button.setStyleSheet("font-size: 18px;")
        self.back_button.clicked.connect(self.show_main_screen)

        self.settings_label = QLabel("Settings Page", self.settings_screen)
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.settings_label.setStyleSheet("font-size: 32px; font-weight: bold;")

        self.style_selector = QComboBox(self.settings_screen)
        self.style_selector.setGeometry(200, 200, 400, 50)
        self.style_selector.setStyleSheet("font-size: 20px;")
        self.style_selector.addItems(["Style 1: Split View", "Style 2: Full Processed View"])
        self.style_selector.currentIndexChanged.connect(self.change_style)

        settings_layout = QVBoxLayout(self.settings_screen)
        settings_layout.addWidget(self.settings_label)
        settings_layout.addWidget(self.style_selector)
        settings_layout.addWidget(self.back_button)
        self.stack.addWidget(self.settings_screen)

        self.current_style = 0  # Default to Style 1

        self.thread = VideoProcessingThread(weights=weights, source=source)
        self.thread.update_original_frame.connect(self.update_original_frame)
        self.thread.update_processed_frame.connect(self.update_processed_frame)
        self.thread.update_status.connect(self.update_status)
        self.thread.start()

    def resizeEvent(self, event):
        if self.stack.currentWidget() == self.main_screen:
            self.splitter.setGeometry(0, 0, self.width(), self.height() - 100)
            self.status_label.setGeometry(0, self.height() - 100, self.width(), 100)
            self.settings_button.setGeometry(self.width() - 150, 10, 140, 50)

    def update_original_frame(self, frame):
        if self.current_style == 0:  # Split View
            pixmap = QPixmap.fromImage(frame)
            scaled_pixmap = pixmap.scaled(self.original_label.width(), self.original_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled_pixmap)

    def update_processed_frame(self, frame):
        pixmap = QPixmap.fromImage(frame)
        if self.current_style == 0:  # Split View
            scaled_pixmap = pixmap.scaled(self.processed_label.width(), self.processed_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_label.setPixmap(scaled_pixmap)
        elif self.current_style == 1:  # Full Processed View
            scaled_pixmap = pixmap.scaled(self.width(), self.height() - 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_label.setPixmap(scaled_pixmap)

    def update_status(self, status, color):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color}; background-color: lightgray;")

    def change_style(self, index):
        self.current_style = index
        if index == 1:  # Full Processed View
            self.splitter.widget(0).hide()
            self.splitter.widget(1).show()
            self.processed_label.setAlignment(Qt.AlignCenter)
        else:  # Split View
            self.splitter.widget(0).show()
            self.splitter.widget(1).show()
            self.original_label.setAlignment(Qt.AlignCenter)
            self.processed_label.setAlignment(Qt.AlignCenter)

    def show_main_screen(self):
        self.stack.setCurrentWidget(self.main_screen)

    def show_settings_screen(self):
        self.stack.setCurrentWidget(self.settings_screen)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    weights_path = "weights/best.pt"
    video_path = "input_video.mp4"
    window = MainApp(weights=weights_path, source=video_path)
    window.show()
    sys.exit(app.exec_())
