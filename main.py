import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QStackedWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
import cv2
import torch
import time
from collections import defaultdict
from coreYoloV5.utils.torch_utils import select_device
from coreYoloV5.utils.dataloaders import LoadImages
from coreYoloV5.utils.general import non_max_suppression, scale_boxes
from coreYoloV5.utils.plots import Annotator, colors
from coreYoloV5.models.common import DetectMultiBackend


class VideoProcessingThread(QThread):
    update_processed_frame = pyqtSignal(QImage)
    update_status = pyqtSignal(str, str)
    update_fps = pyqtSignal(str)

    def __init__(self, weights, source, device=""):
        super().__init__()
        self.weights = weights
        self.source = source
        self.device = device
        self.running = True
        self.tracked_teeth = {}
        self.next_teeth_id = 1

    def run(self):
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device)
        stride, names = model.stride, model.names

        dataset = LoadImages(self.source, img_size=(320, 320), stride=stride)
        prev_time = time.time()

        for path, img, im0s, _, _ in dataset:
            if not self.running:
                break

            start_time = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)
            pred = non_max_suppression(pred, 0.70, 0.45)

            im0_processed = im0s[0].copy() if isinstance(im0s, list) else im0s
            annotator = Annotator(im0_processed, line_width=2)

            teeth_count = 0
            bucket_detected = False
            bucket_size = 0
            current_teeth_positions = []

            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0_processed.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        if conf >= 0.70:
                            x1, y1, x2, y2 = map(int, xyxy)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            if names[int(cls)] == "bucket":
                                bucket_size = (x2 - x1) * (y2 - y1)
                                bucket_detected = True

                            if names[int(cls)] == "teeth":
                                current_teeth_positions.append((center_x, center_y, x1, y1, x2, y2))

            # Tracking teeth positions
            updated_tracked_teeth = {}
            for cx, cy, x1, y1, x2, y2 in current_teeth_positions:
                matched_id = None
                min_distance = float("inf")
                for tid, (px, py) in self.tracked_teeth.items():
                    distance = (cx - px) ** 2 + (cy - py) ** 2
                    if distance < min_distance and distance < 24000:
                        min_distance = distance
                        matched_id = tid

                if matched_id is None:
                    matched_id = self.next_teeth_id
                    self.next_teeth_id += 1

                updated_tracked_teeth[matched_id] = (cx, cy)
                annotator.text((cx, cy), str(matched_id), txt_color=(0, 0, 0))

            self.tracked_teeth = updated_tracked_teeth

            BUCKET_THRESHOLD = 240000  # مقدار قابل تنظیم بسته به ابعاد تصویر (مساحت کادر bucket)

            if bucket_detected and bucket_size >= BUCKET_THRESHOLD:
                if len(updated_tracked_teeth) >= 5:
                    status_text = "سلامت ماشین"
                    status_color = "green"
                else:
                    status_text = "اخطار بررسی سلامت ماشین"
                    status_color = "red"
            else:
                status_text = "سیستم فعال"
                status_color = "blue"

            self.update_status.emit(status_text, status_color)

            im0_processed = annotator.result()

            rgb_processed = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_processed.shape
            bytes_per_line = ch * w
            qt_processed = QImage(rgb_processed.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_processed_frame.emit(qt_processed)

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            self.update_fps.emit(f"FPS: {fps:.2f}")

    def stop(self):
        self.running = False
        self.quit()


class MainApp(QMainWindow):
    def __init__(self, weights, source):
        super().__init__()
        self.setWindowTitle("Shovel")
        self.setGeometry(0, 0, QApplication.desktop().screenGeometry().width(),
                         QApplication.desktop().screenGeometry().height())
        self.setWindowState(Qt.WindowFullScreen)

        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.main_screen = QWidget()

        self.processed_label = QLabel(self.main_screen)
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setGeometry(0, 0, self.width(), self.height() - 100)

        self.fps_label = QLabel(self.main_screen)
        self.fps_label.setGeometry(10, 10, 200, 50)
        self.fps_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: white; background-color: black; padding: 5px;")

        self.status_label = QLabel(self.main_screen)
        self.status_label.setGeometry(0, self.height() - 100, self.width(), 100)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 32px; font-weight: bold; background-color: lightgray;")

        self.stack.addWidget(self.main_screen)

        self.thread = VideoProcessingThread(weights=weights, source=source)
        self.thread.update_processed_frame.connect(self.update_processed_frame)
        self.thread.update_status.connect(self.update_status)
        self.thread.update_fps.connect(self.update_fps)
        self.thread.start()

    def update_processed_frame(self, frame):
        pixmap = QPixmap.fromImage(frame)
        self.processed_label.setPixmap(pixmap)

    def update_status(self, status, color):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"font-size: 32px; font-weight: bold; color: {color}; background-color: lightgray;")

    def update_fps(self, fps_text):
        self.fps_label.setText(fps_text)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    weights_path = "weights/best_new_dataset.pt"
    video_path = "output_2025-01-31_11-48-58.mp4"
    window = MainApp(weights=weights_path, source=video_path)
    window.show()
    sys.exit(app.exec_())
