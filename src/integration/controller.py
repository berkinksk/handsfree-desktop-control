"""
System integrator stub using PyAutoGUI later.
"""
import os
import sqlite3

class HeadEyeController:
    def __init__(self):
        # Initialize database connection
        db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'db'))
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, 'settings.db')
        self.conn = sqlite3.connect(self.db_path)
        self._initialize_database()

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
        print("TODO calibrate")  # will save neutral pose in DB
        return True

    def start_control(self):
        print("TODO start control loop")

    def stop_control(self):
        print("TODO stop control loop") 