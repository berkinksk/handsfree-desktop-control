import cv2
import numpy as np
import math
import os

class HeadEyeDetector:
    """Detect head pose and blinks from video frames."""
    def __init__(self):
        # Initialize face detector (Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            print("Warning: Haar cascade not loaded; face detection will be disabled.")
        # Initialize facemark model for 68 facial landmarks
        self.landmark_model = None
        try:
            self.landmark_model = cv2.face.createFacemarkLBF()
            # Path to the pretrained LBF model file (make sure it exists)
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lbfmodel.yaml')
            self.landmark_model.loadModel(model_path)
        except Exception as e:
            print(f"Warning: failed to load facemark model: {e}")
            self.landmark_model = None

        # Define 3D model points of facial landmarks for pose estimation.
        # These points correspond to the 2D landmark indices:
        # 0: nose tip, 1: chin, 2: left eye corner, 3: right eye corner, 4: left mouth, 5: right mouth.
        # The 3D coordinates are an approximate model of a generic face (in mm).
        self.model_points_3D = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -150.0, -150.0),  # Chin (roughly 15cm below nose tip, back 15cm)
            (-150.0, 150.0, -125.0),# Left eye left corner (x=-15cm, y=15cm, z=-12.5cm)
            (150.0, 150.0, -125.0), # Right eye right corner (x=15cm, y=15cm, z=-12.5cm)
            (-100.0, -150.0, -125.0), # Left mouth corner (x=-10cm, y=-15cm, z=-12.5cm)
            (100.0, -150.0, -125.0)   # Right mouth corner (x=10cm, y=-15cm, z=-12.5cm)
        ], dtype=np.float32)

        # We will set camera calibration parameters when we know the frame size (done in detect_head_pose).
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion

    def detect_face(self, frame):
        """Detect a face in the frame. Returns (x, y, w, h) for the first face or None if not found."""
        if self.face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image (scaleFactor and minNeighbors tuned for webcam face)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        # Pick the first face detected (faces is an array of [x,y,w,h] boxes)
        return faces[0]

    def detect_landmarks(self, frame, face_box):
        """Detect 68 facial landmarks within the given face box. Returns a list of (x,y) points."""
        if self.landmark_model is None or face_box is None:
            return []
        # Prepare the face region for landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # OpenCV Facemark LBF expects a list of rectangles [ (x,y,w,h) ], here we pass our single face box.
        ok, landmarks = self.landmark_model.fit(gray, [tuple(face_box)])
        if not ok or landmarks is None:
            return []
        # landmarks[0] is an array of shape (68, 2) for the 68 points
        points = landmarks[0][0] if isinstance(landmarks[0], tuple) else landmarks[0]
        # Convert to Python list of (x, y) tuples
        return [(int(pt[0]), int(pt[1])) for pt in points]

    def detect_head_pose(self, frame):
        """Estimate the head pose as one of: "left", "right", "up", "down", "center".""" 
        # Step 1: Face detection
        face = self.detect_face(frame)
        if face is None:
            # No face detected, can't determine pose (return "center" as a fallback)
            return "center"
        x, y, w, h = face
        # Step 2: Facial landmark detection
        landmarks = self.detect_landmarks(frame, (x, y, w, h))
        if len(landmarks) < 68:
            # Landmarks not detected, return center as default
            return "center"
        # Define the 2D image points from landmark positions
        image_points = np.array([
            landmarks[30],  # Nose tip (index 30 in 0-based index)
            landmarks[8],   # Chin (index 8)
            landmarks[36],  # Left eye left corner (index 36)
            landmarks[45],  # Right eye right corner (index 45)
            landmarks[48],  # Left mouth corner (index 48)
            landmarks[54]   # Right mouth corner (index 54)
        ], dtype=np.float32)
        # Initialize camera matrix if not done yet or if frame size changed
        if self.camera_matrix is None:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            focal_length = frame_width  # approximate focal length using frame width
            center = (frame_width / 2, frame_height / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
        # Step 3: SolvePnP to find 3D pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3D, image_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            # solvePnP failed (should not usually happen if points are valid)
            return "center"
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # Project the rotation matrix to Euler angles (in degrees) for yaw, pitch, and roll
        # Using the conventions: 
        # yaw   = rotation around Y-axis (left/right),
        # pitch = rotation around X-axis (up/down),
        # roll  = rotation around Z-axis (tilt sideways).
        # We derive yaw, pitch from the rotation matrix.
        # Avoid gimbal lock by checking for cos(pitch) near zero.
        sy = math.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch_rad = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
            yaw_rad   = math.atan2(-rotation_matrix[2,0], sy)
            roll_rad  = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            # Gimbal lock case (pitch ~ 90°)
            pitch_rad = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            yaw_rad   = math.atan2(-rotation_matrix[2,0], sy)
            roll_rad  = 0
        # Convert to degrees
        pitch = math.degrees(pitch_rad)
        yaw   = math.degrees(yaw_rad)
        roll  = math.degrees(roll_rad)
        # Step 4: Determine direction based on yaw/pitch angles
        direction = "center"
        # We set threshold angles to decide what is considered a significant turn/tilt
        YAW_THRESHOLD = 20    # degrees left/right
        PITCH_THRESHOLD = 15  # degrees up/down
        if yaw < -YAW_THRESHOLD:
            direction = "right"   # (Note: if yaw is negative, user turned right from camera perspective)
        elif yaw > YAW_THRESHOLD:
            direction = "left"
        elif pitch < -PITCH_THRESHOLD:
            direction = "up"      # negative pitch: looking up (camera sees chin)
        elif pitch > PITCH_THRESHOLD:
            direction = "down"    # positive pitch: looking down (camera sees forehead)
        else:
            direction = "center"
        return direction

    def detect_blink(self, frame):
        """Return True if a blink (eyes closed) is detected in the frame, else False."""
        # For now, this is a placeholder. We could implement a simple eye-aspect-ratio check using landmarks,
        # or integrate a CNN model in the future.
        return False

    def test_camera(self):
        """Test camera connectivity by capturing one frame and printing its dimensions."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return False
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("Error: Cannot read frame from webcam")
            return False
        print(f"Captured frame size: {frame.shape[1]}x{frame.shape[0]}")
        return True
