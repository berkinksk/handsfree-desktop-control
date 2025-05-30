"""
System integrator using PyAutoGUI for cursor control.
Maps head pose to cursor movement and blinks to clicks.
"""
import pyautogui
import time
from PyQt5 import QtCore, QtGui # QtGui might be needed for QImage if we pass pixmaps later
import cv2 # For camera capture
import numpy as np # For frame manipulation

# Assuming backend.detector is in PYTHONPATH or src is structured as a package
from backend.detector import HeadEyeDetector

class CursorController:
    """Controls the mouse cursor based on head pose and eye actions."""
    def __init__(self, screen_width, screen_height, 
                 horizontal_sensitivity=20, vertical_sensitivity=20, 
                 smoothing_factor=0.5, dead_zone_x=0.5, dead_zone_y=0.5):
        """
        Initializes the CursorController.

        Args:
            screen_width (int): Width of the screen in pixels.
            screen_height (int): Height of the screen in pixels.
            horizontal_sensitivity (float): Factor to multiply yaw by for horizontal movement.
            vertical_sensitivity (float): Factor to multiply pitch by for vertical movement.
            smoothing_factor (float): Alpha for exponential moving average for cursor movement (0-1). Higher means more smoothing.
            dead_zone_x (float): Absolute yaw value below which no horizontal movement occurs.
            dead_zone_y (float): Absolute pitch value below which no vertical movement occurs.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.horizontal_sensitivity = horizontal_sensitivity
        self.vertical_sensitivity = vertical_sensitivity
        self.smoothing_factor = smoothing_factor
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y

        self.current_x = screen_width // 2
        self.current_y = screen_height // 2
        pyautogui.FAILSAFE = False # Disable failsafe for smoother control (use with caution)
        pyautogui.PAUSE = 0 # No pause between pyautogui calls

        # Initial mouse position to center
        pyautogui.moveTo(self.current_x, self.current_y)

    def update_cursor(self, yaw, pitch, pose_label, action_detected):
        """
        Updates the cursor position based on yaw and pitch, and performs a click if action_detected.

        Args:
            yaw (float): Smoothed yaw angle from the HeadEyeDetector.
            pitch (float): Smoothed pitch angle from the HeadEyeDetector.
            pose_label (str): The current detected pose (e.g., "center", "left", etc.).
            action_detected (bool): True if the eye action (e.g., pupil-based) was detected.
        """
        
        dx = 0
        dy = 0

        # Apply dead zone & sensitivity for X movement
        if abs(yaw) > self.dead_zone_x:
            # Invert yaw for natural cursor movement (head left -> cursor left)
            dx = -yaw * self.horizontal_sensitivity 
        
        # Apply dead zone & sensitivity for Y movement
        if abs(pitch) > self.dead_zone_y:
            # Detector: positive pitch = head down, negative pitch = head up.
            # Screen Y: increases downwards.
            # To correct observed mirrored movement (head up -> cursor down, head down -> cursor up):
            # we invert the relationship. If positive pitch (head down) should move cursor UP (negative dy),
            # and negative pitch (head up) should move cursor DOWN (positive dy), then we need -pitch.
            dy = -pitch * self.vertical_sensitivity # Inverted pitch contribution

        # Apply smoothing to target position
        target_x = self.current_x + dx
        target_y = self.current_y + dy

        self.current_x = (1 - self.smoothing_factor) * target_x + self.smoothing_factor * self.current_x
        self.current_y = (1 - self.smoothing_factor) * target_y + self.smoothing_factor * self.current_y
        
        # Clamp to screen boundaries
        self.current_x = max(0, min(self.screen_width - 1, self.current_x))
        self.current_y = max(0, min(self.screen_height - 1, self.current_y))

        pyautogui.moveTo(int(self.current_x), int(self.current_y), duration=0) # Move instantly

        if action_detected:
            print(f"Controller: Action detected! Performing click at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.click()
            # Detector should handle action reset logic to prevent multiple clicks for a single held action.
            time.sleep(0.1) # Small delay to prevent immediate re-click if action is still held by detector logic

    @staticmethod
    def get_screen_resolution():
        """Gets the primary screen resolution."""
        return pyautogui.size()

# Placeholder for GUI integration, to be fully implemented as per Step 4 of Improvement Plan
# from PyQt5 import QtCore # Already imported at top

class HeadEyeController(QtCore.QObject):
    """ 
    Orchestrates head/eye detection and cursor control.
    This is the main controller the GUI interacts with.
    """
    blink_detected = QtCore.pyqtSignal() # Signal for blink/action events
    frame_processed = QtCore.pyqtSignal(np.ndarray) # Signal to send processed frame (RGB) to GUI
    calibration_status = QtCore.pyqtSignal(str) # Signal for calibration status updates
    error_occurred = QtCore.pyqtSignal(str) # Signal for error messages

    PROCESSING_INTERVAL_MS = 30 # approx 33 FPS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.calibrated_center_yaw = 0.0
        self.calibrated_center_pitch = 0.0
        
        try:
            self.detector = HeadEyeDetector() # Use default thresholds from detector
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize HeadEyeDetector: {e}")
            self.detector = None # Ensure detector is None if init fails
            return # Stop initialization if detector fails

        screen_w, screen_h = CursorController.get_screen_resolution()
        # TODO: Make sensitivities, smoothing, dead_zone configurable via GUI
        self.cursor_controller = CursorController(
            screen_w, screen_h,
            horizontal_sensitivity=10, # Default value, halved from 20
            vertical_sensitivity=12, # Default value, halved from 25
            smoothing_factor=0.6,    # Default value
            dead_zone_x=1.0,         # Default value
            dead_zone_y=1.0          # Default value
        )

        self.capture = None
        self.processing_timer = QtCore.QTimer(self)
        self.processing_timer.timeout.connect(self._process_next_frame)

    def set_cursor_sensitivity(self, horizontal: int, vertical: int):
        """Allows GUI to set cursor sensitivity."""
        if self.cursor_controller:
            self.cursor_controller.horizontal_sensitivity = horizontal
            self.cursor_controller.vertical_sensitivity = vertical
            print(f"Cursor sensitivity updated: H={horizontal}, V={vertical}")

    def calibrate(self) -> bool:
        if not self.detector or not self.running or not self.capture or not self.capture.isOpened():
            self.calibration_status.emit("Calibration Failed: Control not running or camera issue.")
            return False
        
        ret, frame = self.capture.read()
        if not ret:
            self.calibration_status.emit("Calibration Failed: Could not read frame.")
            return False

        # Flip frame like in GUI for consistency if needed by user, but detector works on raw frame
        # frame_for_detection = cv2.flip(frame, 1) 
        
        detection_results = self.detector.process_frame_and_detect_features(frame)

        if detection_results and detection_results["landmarks_mp_object"]:
            self.calibrated_center_yaw = detection_results["raw_yaw"]
            self.calibrated_center_pitch = detection_results["raw_pitch"]
            self.calibration_status.emit(f"Calibrated: Center Yaw={self.calibrated_center_yaw:.2f}, Pitch={self.calibrated_center_pitch:.2f}")
            print(f"HeadEyeController: Calibrated. Center Yaw: {self.calibrated_center_yaw}, Pitch: {self.calibrated_center_pitch}")
            return True
        else:
            self.calibration_status.emit("Calibration Failed: No face detected.")
            print("HeadEyeController: Calibration failed, no face detected.")
            return False

    def start_control(self):
        if not self.detector:
            self.error_occurred.emit("Cannot start: HeadEyeDetector not initialized.")
            return

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW for Windows for better performance/compatibility
        if not self.capture.isOpened():
            self.error_occurred.emit("Error: Webcam not accessible.")
            self.capture = None
            return

        self.running = True
        self.processing_timer.start(self.PROCESSING_INTERVAL_MS)
        print("HeadEyeController: Start control called, timer started.")

    def stop_control(self):
        self.running = False
        self.processing_timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        print("HeadEyeController: Stop control called, timer stopped, camera released.")

    def _process_next_frame(self):
        if not self.running or not self.capture or not self.capture.isOpened() or not self.detector:
            return

        ret, frame = self.capture.read()
        if not ret:
            print("HeadEyeController: Failed to capture frame.")
            return

        # It's common practice to flip the frame horizontally for a mirror effect in GUIs
        # The detector should ideally work with the raw camera feed if its PnP model assumes that,
        # or the flip should be consistent with how landmarks were chosen/model trained.
        # For now, let's assume detector handles un-flipped frames.
        # We'll flip it *after* detection only for display if needed.
        
        detection_results = self.detector.process_frame_and_detect_features(frame.copy()) # Pass a copy

        if detection_results:
            # Adjust yaw and pitch based on calibration
            current_yaw = detection_results["raw_yaw"] - self.calibrated_center_yaw
            current_pitch = detection_results["raw_pitch"] - self.calibrated_center_pitch
            
            self.cursor_controller.update_cursor(
                current_yaw,
                current_pitch,
                detection_results["pose"],
                detection_results["action_detected"]
            )
            if detection_results["action_detected"]:
                self.blink_detected.emit()

        # Prepare frame for GUI (convert BGR to RGB and flip for mirror view)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped_rgb_frame = cv2.flip(rgb_frame, 1)
        self.frame_processed.emit(flipped_rgb_frame)

# Example Usage (for testing this module directly)
# This part is usually commented out or removed when integrating
# if __name__ == '__main__':
# try:
# screen_w, screen_h = CursorController.get_screen_resolution()
# print(f"Screen Resolution: {screen_w}x{screen_h}")
# 
# controller = CursorController(
# screen_w, screen_h, 
# horizontal_sensitivity=25,
# vertical_sensitivity=35,
# smoothing_factor=0.6,
# dead_zone_x=1.0,
# dead_zone_y=1.0
# )
# 
# print("Starting dummy cursor control test...")
# print("Move your head (conceptually) to test. Action triggers click.")
# print("Press Ctrl+C to exit.")
# 
# test_inputs = [
#             (0, 0, "center", False),
#             (-2, 0, "left", False),
#             (-5, 0, "left", False),
#             (0, 0, "center", False),
#             (2, 0, "right", False),
#             (5, 0, "right", True),
#             (0, 0, "center", False),
#             (0, -2, "up", False), 
#             (0, -4, "up", False),
#             (0, 0, "center", False),
#             (0, 2, "down", False), 
#             (0, 4, "down", True),
#             (0, 0, "center", False),
#             (-5, -4, "left_up", True),
#             (0,0, "center", False)
# ]
# 
# for yaw, pitch, pose, action in test_inputs:
# print(f"Input: Yaw={yaw}, Pitch={pitch}, Pose={pose}, Action={action}")
# controller.update_cursor(yaw, pitch, pose, action)
# time.sleep(0.75)
# 
# print("Dummy test finished. Returning mouse to center.")
# controller.update_cursor(0,0,"center", False)
# 
# except KeyboardInterrupt:
# print("\nCursor control test stopped by user.")
# except Exception as e:
# print(f"An error occurred: {e}")
#
# To test HeadEyeController standalone (without full GUI):
# if __name__ == '__main__':
#     from PyQt5.QtWidgets import QApplication
#     import sys
# 
#     app = QApplication(sys.argv) # QApplication instance is needed for QTimer
#     he_controller = HeadEyeController()
# 
#     def on_blink():
#         print("EVENT: Blink detected by HeadEyeController!")
# 
#     def on_frame(cv_frame):
#         # In a real app, you'd display this in a QLabel
#         # For testing, just show with OpenCV
#         cv2.imshow("HeadEyeController Test", cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             he_controller.stop_control()
#             cv2.destroyAllWindows()
#             app.quit()
# 
#     def on_calib_status(status):
#         print(f"CALIBRATION: {status}")
# 
#     def on_error(errmsg):
#         print(f"ERROR: {errmsg}")
#         he_controller.stop_control()
#         app.quit()
# 
#     if not he_controller.detector: # Check if detector failed to init
#         sys.exit(1)
# 
#     he_controller.blink_detected.connect(on_blink)
#     he_controller.frame_processed.connect(on_frame)
#     he_controller.calibration_status.connect(on_calib_status)
#     he_controller.error_occurred.connect(on_error)
# 
#     he_controller.start_control()
# 
#     # Simple calibration test after 5 seconds
#     QtCore.QTimer.singleShot(5000, lambda: he_controller.calibrate())
# 
#     print("Running HeadEyeController test... Press 'q' in OpenCV window to quit.")
#     sys.exit(app.exec_()) 