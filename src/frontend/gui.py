from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Head-Eye Control System")
        self.resize(800, 600)
        central = QWidget()
        layout = QVBoxLayout(central)

        lbl = QLabel("Stub GUI – functionality coming soon")
        btnStart = QPushButton("Start")
        btnStart.setEnabled(False)
        btnStop = QPushButton("Stop")
        btnStop.setEnabled(False)
        btnCal = QPushButton("Calibrate")
        btnCal.setEnabled(False)

        layout.addWidget(lbl)
        layout.addWidget(btnStart)
        layout.addWidget(btnStop)
        layout.addWidget(btnCal)
        self.setCentralWidget(central)

def launch_app():
    """Launch the stub GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    launch_app() 