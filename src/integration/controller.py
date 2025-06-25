import pyautogui
import time
from PyQt5 import QtCore, QtGui # QtGui might be needed for QImage if we pass pixmaps later
import cv2 # For camera capture
import numpy as np # For frame manipulation

# Assuming backend.detector is in PYTHONPATH or src is structured as a package
from backend.detector import HeadEyeDetector

class CursorController:
    """Controls the mouse cursor based on head pose and eye actions."""

    # Define action type strings recognized by the controller
    # These should match the strings returned by HeadEyeDetector's process_frame_and_detect_features
    CLICK_TYPE_NONE = "NO_ACTION"
    CLICK_TYPE_SINGLE = "SINGLE_CLICK"  # Generic single click (e.g., for non-center poses)
    CLICK_TYPE_DOUBLE = "DOUBLE_CLICK"  # Generic double click (e.g., for non-center poses)
    CLICK_TYPE_LEFT_SINGLE = "LEFT_SINGLE_CLICK"
    CLICK_TYPE_LEFT_DOUBLE = "LEFT_DOUBLE_CLICK"
    CLICK_TYPE_RIGHT_SINGLE = "RIGHT_SINGLE_CLICK"
    # CLICK_TYPE_RIGHT_DOUBLE is intentionally omitted as it's not used.

    def __init__(self, screen_width, screen_height, 
                 horizontal_sensitivity=20, vertical_sensitivity=20, 
                 smoothing_factor=0.5, dead_zone_x=0.5, dead_zone_y=0.5):
        
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

    def update_cursor(self, yaw, pitch, pose_label, action_type):
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

        # Access click type constants from the detector instance if possible, or define them here.
        # For now, using string literals as passed by the modified detector.
        
        # Define action type strings (could also be imported from detector if they become more complex)
        # CLICK_TYPE_SINGLE = "SINGLE_CLICK" 
        # CLICK_TYPE_LEFT_SINGLE = "LEFT_SINGLE_CLICK"
        # CLICK_TYPE_LEFT_DOUBLE = "LEFT_DOUBLE_CLICK"
        # CLICK_TYPE_RIGHT_SINGLE = "RIGHT_SINGLE_CLICK"

        if action_type == self.CLICK_TYPE_LEFT_SINGLE:
            print(f"Controller: LEFT_SINGLE_CLICK detected! Performing left click at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.click()
        elif action_type == self.CLICK_TYPE_LEFT_DOUBLE:
            print(f"Controller: LEFT_DOUBLE_CLICK detected! Performing left double click at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.doubleClick()
        elif action_type == self.CLICK_TYPE_RIGHT_SINGLE:
            print(f"Controller: RIGHT_SINGLE_CLICK detected! Performing right click (context menu) at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.rightClick()
        elif action_type == self.CLICK_TYPE_SINGLE: # Generic single click (from non-center poses)
            print(f"Controller: GENERIC SINGLE_CLICK detected! Performing left click at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.click()
        elif action_type == self.CLICK_TYPE_DOUBLE: # Generic double click (from non-center poses)
            print(f"Controller: GENERIC DOUBLE_CLICK detected! Performing left double click at ({int(self.current_x)}, {int(self.current_y)})")
            pyautogui.doubleClick()
        # elif action_type == "DOUBLE_CLICK": # Check for double click # This was the old generic one
        #     print(f"Controller: DOUBLE_CLICK detected! Performing double click at ({int(self.current_x)}, {int(self.current_y)})")
        #     pyautogui.doubleClick()
        #     # time.sleep(0.1) # Original delay

    @staticmethod
    def get_screen_resolution():
        """Gets the primary screen resolution."""
        return pyautogui.size()

class HeadEyeController(QtCore.QObject):
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
            
            # Use the new action_type key from detection_results
            action_type_from_detector = detection_results.get("action_type", self.detector.CLICK_TYPE_NONE) 
            
            print(f"CONTROLLER DEBUG: Pose={detection_results['pose']}, ActionType={action_type_from_detector}, NormPupilYDiff={detection_results['norm_pupil_y_diff']:.3f}")

            self.cursor_controller.update_cursor(
                current_yaw,
                current_pitch,
                detection_results["pose"],
                action_type_from_detector # Pass the action_type
            )
            if action_type_from_detector != self.detector.CLICK_TYPE_NONE: # Emit signal if any click occurred
                self.blink_detected.emit()

        # Prepare frame for GUI (convert BGR to RGB and flip for mirror view)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped_rgb_frame = cv2.flip(rgb_frame, 1)
        self.frame_processed.emit(flipped_rgb_frame)
