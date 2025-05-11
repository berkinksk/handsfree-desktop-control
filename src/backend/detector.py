"""
Stubbed vision backend.

class HeadEyeDetector:
    detect_head_pose(frame) -> str
        returns one of {"left","right","up","down","center"}
    detect_blink(frame) -> bool
        True if (placeholder) blink detected
"""
import cv2
import os
import ctypes
import ctypes.wintypes
import numpy as np
import math

class HeadEyeDetector:
    """Detect head pose and blinks from video frames."""
    def __init__(self):
        # helper to get Windows short path for Unicode support
        def _get_short_path(path):
            if os.name == 'nt':
                buf = ctypes.create_unicode_buffer(260)
                ctypes.windll.kernel32.GetShortPathNameW(path, buf, ctypes.sizeof(buf))
                return buf.value
            return path
        # load Haar cascade for face detection from models folder
        cascade_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'haarcascade_frontalface_default.xml'))
        cascade_path = _get_short_path(cascade_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print(f"Warning: cascade not loaded from {cascade_path}; face detection will be disabled.")
        # load LBF facemark model for landmarks
        self.landmark_model = None
        try:
            # create facemark and load model file
            self.landmark_model = cv2.face.createFacemarkLBF()
            # point to the project-level models directory
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lbfmodel.yaml'))
            model_path = _get_short_path(model_path)
            self.landmark_model.loadModel(model_path)
        except Exception as e:
            print(f"Warning: failed to load facemark model: {e}")
            self.landmark_model = None

    def detect_head_pose(self, frame):
        """Estimate head orientation and return one of: left, right, up, down, center."""
        # detect face and landmarks
        face = self.detect_face(frame)
        landmarks = self.detect_landmarks(frame, face)
        # need at least six points for solvePnP
        if face is None or not landmarks or len(landmarks) < 55:
            return "center"

        # Image points from landmarks
        image_pts = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype="double")
        # 3D model points
        model_pts = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype="double")
        # Camera parameters
        h, w = frame.shape[:2]
        focal_len = w
        cam_mat = np.array([[focal_len, 0, w/2],
                            [0, focal_len, h/2],
                            [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(model_pts, image_pts, cam_mat, dist_coeffs)
        if not success:
            return "center"
        # Rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        # Euler angles
        sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2,1], rmat[2,2])
            y = math.atan2(-rmat[2,0], sy)
            z = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], sy)
            z = 0
        # Convert radians to degrees
        pitch, yaw, roll = np.degrees([x, y, z])

        # Threshold to determine direction
        thresh = 15.0
        if yaw > thresh:
            return "right"
        elif yaw < -thresh:
            return "left"
        elif pitch > thresh:
            return "down"
        elif pitch < -thresh:
            return "up"
        return "center"

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

    def detect_landmarks(self, frame, face_box):
        """Detect facial landmarks using pretrained LBF model; returns list of (x,y) or empty."""
        # ensure model is loaded and face_box is provided
        if self.landmark_model is None or face_box is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Facemark expects face rectangles as a numpy array of shape (N,4)
        rects = np.array([face_box], dtype=np.int32)
        ok, landmarks = self.landmark_model.fit(gray, rects)
        if not ok or not landmarks:
            return []
        # landmarks[0] is N x 2 array of points
        return landmarks[0].tolist()

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