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
        # load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            print("Warning: Haar cascade not loaded; face detection will be disabled.")

    def detect_head_pose(self, frame):
        """Return head pose as one of: left, right, up, down, center."""
        return "center"  # TODO replace with real logic

    def detect_blink(self, frame):
        """Return True if placeholder blink detected."""
        return False  # TODO replace with real logic

    def detect_face(self, frame):
        """Detect a face in the frame and return its bounding box (x, y, w, h) or None."""
        # if cascade failed to load, skip detection
        if self.face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        # return the first detected face
        return faces[0]

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