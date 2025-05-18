"""
System integrator stub using PyAutoGUI later.
"""
import os
import sqlite3
import threading
import time
import cv2
from backend.detector import HeadEyeDetector

class HeadEyeController:
    def __init__(self):
        # Initialize database connection
        db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'db'))
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, 'settings.db')
        self.conn = sqlite3.connect(self.db_path)
        self._initialize_database()

        # Initialize vision backend processor
        self.detector = HeadEyeDetector()
        self.running = False

    def _initialize_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                neutral_yaw REAL NOT NULL,
                neutral_pitch REAL NOT NULL,
                neutral_roll REAL NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        )
        self.conn.commit()

    def calibrate(self):
        # Calibration routine: capture head pose data for 2 seconds to determine neutral position
        print("Calibration started")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam for calibration")
            return False
        start_time = time.time()
        yaw_vals, pitch_vals = [], []
        while time.time() - start_time < 2.0:
            ret, frame = cap.read()
            if not ret:
                continue
            pose = self.detector.detect_head_pose(frame)
            # Map pose string to numeric yaw and pitch
            y = -1.0 if pose == "left" else (1.0 if pose == "right" else 0.0)
            p = 1.0 if pose == "up" else (-1.0 if pose == "down" else 0.0)
            yaw_vals.append(y)
            pitch_vals.append(p)
        cap.release()
        # Compute neutral positions
        neutral_yaw = sum(yaw_vals) / len(yaw_vals) if yaw_vals else 0.0
        neutral_pitch = sum(pitch_vals) / len(pitch_vals) if pitch_vals else 0.0
        neutral_roll = 0.0
        # Store in database
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO calibration (neutral_yaw, neutral_pitch, neutral_roll) VALUES (?, ?, ?)",
            (neutral_yaw, neutral_pitch, neutral_roll)
        )
        self.conn.commit()
        print(f"Calibration complete: neutral_yaw={neutral_yaw}, neutral_pitch={neutral_pitch}, neutral_roll={neutral_roll}")
        return True

    def start_control(self):
        # Start the continuous capture loop for head pose and blink detection
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()

    def stop_control(self):
        # Stop the capture loop
        if not self.running:
            return
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def _control_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            pose = self.detector.detect_head_pose(frame)
            blink = self.detector.detect_blink(frame)
            print(f"Head Pose: {pose}, Blink: {blink}")
            time.sleep(0.1)
        cap.release() 