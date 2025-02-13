import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class ReportsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 گزارشات زنده")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # نمودار FPS
        self.fps_plot = pg.PlotWidget()
        self.fps_plot.setBackground("#121212")
        self.fps_plot.setTitle("نوسان FPS", color="w", size="15pt")
        self.fps_plot.setLabel("left", "FPS", color="w")
        self.fps_plot.setLabel("bottom", "Frame", color="w")
        self.fps_curve = self.fps_plot.plot(pen=pg.mkPen(color="cyan", width=2))
        self.fps_data = np.zeros(100)

        layout.addWidget(self.fps_plot)
        self.setLayout(layout)

    def update_fps(self, fps):
        """ بروزرسانی مقدار FPS در نمودار """
        self.fps_data = np.roll(self.fps_data, -1)
        self.fps_data[-1] = fps
        self.fps_curve.setData(self.fps_data)
