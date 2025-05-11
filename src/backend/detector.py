"""
Stubbed vision backend.

class HeadEyeDetector:
    detect_head_pose(frame) -> str
        returns one of {"left","right","up","down","center"}
    detect_blink(frame) -> bool
        True if (placeholder) blink detected
"""
import cv2

class HeadEyeDetector:
    """Detect head pose and blinks from video frames."""
    def __init__(self):
        pass

    def detect_head_pose(self, frame):
        """Return head pose as one of: left, right, up, down, center."""
        return "center"  # TODO replace with real logic

    def detect_blink(self, frame):
        """Return True if placeholder blink detected."""
        return False  # TODO replace with real logic

    def test_camera(self):
        """Test camera setup by capturing one frame and logging its dimensions."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return False
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Cannot read frame from webcam")
            return False
        print(f"Captured frame size: {frame.shape[1]}x{frame.shape[0]}")
        return True 