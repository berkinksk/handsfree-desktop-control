from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import sys

def launch_app():
    """Launch the stub GUI application."""
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Hands-Free Desktop Control")
    central = QWidget()
    layout = QVBoxLayout(central)

    lbl = QLabel("Stub GUI – functionality coming soon")
    btnStart = QPushButton("Start");     btnStart.setEnabled(False)
    btnStop  = QPushButton("Stop");      btnStop.setEnabled(False)
    btnCal   = QPushButton("Calibrate"); btnCal.setEnabled(False)

    layout.addWidget(lbl)
    layout.addWidget(btnStart)
    layout.addWidget(btnStop)
    layout.addWidget(btnCal)
    win.setCentralWidget(central)
    win.resize(400, 200)
    win.show()
    sys.exit(app.exec_()) 