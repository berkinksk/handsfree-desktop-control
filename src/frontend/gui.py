from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QSlider, QSpinBox
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import cv2
from PyQt5.QtGui import QImage, QPixmap
import sys

class MainWindow(QMainWindow):
    # Signals for thread-safe updates
    status_update = pyqtSignal(str)
    blink_detected = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Head-Eye Control System")
        self.resize(800, 600)
        central = QWidget()
        layout = QVBoxLayout(central)

        # Status label
        self.status_label = QLabel("Status: Not Calibrated")
        layout.addWidget(self.status_label)

        # Control buttons
        self.btn_start = QPushButton("Start")
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.setEnabled(True)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_calibrate)

        # Connect the Calibrate button to the stub calibration slot
        self.btn_calibrate.clicked.connect(self.on_calibrate)

        # Status indicators placeholders
        self.tracking_indicator = QLabel("Tracking: Off")
        self.blink_indicator = QLabel("Blink: N/A")
        layout.addWidget(self.tracking_indicator)
        layout.addWidget(self.blink_indicator)

        # Camera preview setup
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        # Initialize camera capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam for preview")

        # Timer for updating camera frames
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_frame)
        self.preview_timer.start(30)

        # Settings controls
        self.sensitivity_label = QLabel("Cursor Sensitivity")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        layout.addWidget(self.sensitivity_label)
        layout.addWidget(self.sensitivity_slider)

        self.blink_threshold_label = QLabel("Blink Detection Threshold")
        self.blink_threshold_spinbox = QSpinBox()
        self.blink_threshold_spinbox.setRange(1, 10)
        self.blink_threshold_spinbox.setValue(5)
        layout.addWidget(self.blink_threshold_label)
        layout.addWidget(self.blink_threshold_spinbox)

        # Connect settings controls to stub handlers
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        self.blink_threshold_spinbox.valueChanged.connect(self.on_blink_threshold_changed)

        self.setCentralWidget(central)
        
        # Connect signals to update slots
        self.status_update.connect(self.update_status_label)
        self.blink_detected.connect(self.on_blink_detected)
        
        # Dummy calls to test real-time updates
        QTimer.singleShot(1000, lambda: self.status_update.emit("Calibrated"))
        QTimer.singleShot(2000, lambda: self.blink_detected.emit())

    def on_calibrate(self):
        """Stub slot for calibration button click."""
        # Update status label and print to console as a stub action
        self.status_label.setText("Status: Calibrating...")
        print("Calibration requested")

    def on_sensitivity_changed(self, value):
        """Stub handler for sensitivity slider value change."""
        print(f"Sensitivity changed to {value}")
        self.sensitivity_label.setText(f"Cursor Sensitivity: {value}")

    def on_blink_threshold_changed(self, value):
        """Stub handler for blink threshold spinbox value change."""
        print(f"Blink threshold changed to {value}")
        self.blink_threshold_label.setText(f"Blink Detection Threshold: {value}")

    def update_status_label(self, status):
        """Slot to update the main status label."""
        self.status_label.setText(f"Status: {status}")
        print(f"Status updated to {status}")

    def on_blink_detected(self):
        """Slot to indicate a blink event in the GUI."""
        # Flash blink indicator label
        self.blink_indicator.setText("Blink: Detected!")
        print("Blink detected event")
        # Reset after short delay
        QTimer.singleShot(500, lambda: self.blink_indicator.setText("Blink: N/A"))

    def update_frame(self):
        """Read frame from camera and display it in the GUI."""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return
            # Convert frame to RGB and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.camera_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Ensure camera resource is released on close."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        event.accept()

def launch_app():
    """Launch the stub GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    launch_app() 

    