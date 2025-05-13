import cv2
import numpy as np
import math
import os
# Initialize Mediapipe within the class to avoid global dependency issues

class HeadEyeDetector:
    """Detect head pose and blinks from video frames using Mediapipe Face Mesh."""
    def __init__(self):
        # Try importing Mediapipe (must be installed in the environment)
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("Mediapipe is not installed. Install it via 'pip install mediapipe'.")
        # Initialize Mediapipe FaceMesh for one face
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,       # iris landmarks not needed for our use
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Blink detection settings
        self.blink_threshold = 0.2          # EAR threshold for blink
        self.blink_consec_frames = 3        # frames required below threshold to count a blink
        self.blink_counter = 0             # frame counter for consecutive blink frames
        # Head pose threshold defaults (in degrees)
        self.pose_thresholds = {"left": 5.0, "right": 5.0, "up": 5.0, "down": 5.0}  # lower thresholds for pose detection
        # Smoothing state
        self.last_pose = "center"
        self.pose_consec_frames = 2        # require 2 consecutive frames to confirm pose change
        self.pose_counter = 0
        # Store last raw angles for debugging/display
        self.last_raw_pitch = 0.0
        self.last_raw_yaw = 0.0
        # Adaptive thresholding state
        self.adaptive_thresholds = False    # disable adaptive thresholding for consistency
        self.pitch_max = 0.0
        self.pitch_min = 0.0
        self.yaw_max = 0.0
        self.angle_history = []
        self.history_size = 20
        self.initial_frames = 0           # counter for initial calibration frames
        # Internal flag and storage for landmark reuse
        self._landmarks_ready = False
        self.last_landmarks = []          # will hold last detected landmarks (list of (x,y) points)

    def detect_face(self, frame):
        """Detect face in the frame and store landmarks. Returns (x,y,w,h) of face or None."""
        # Convert BGR image to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            # Use the first detected face's landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            # Convert normalized landmark coordinates to pixel coordinates
            self.last_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            # Compute a bounding box around the face landmarks
            xs = [pt[0] for pt in self.last_landmarks]; ys = [pt[1] for pt in self.last_landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            face_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            # Flag that landmarks for this frame are ready to use
            self._landmarks_ready = True
            return face_box
        else:
            # No face detected in this frame
            self.last_landmarks = []
            self._landmarks_ready = False
            return None

    def detect_landmarks(self, frame, face_box):
        """Return facial landmarks for the detected face as a list of (x,y) points, or [] if none."""
        # If landmarks from detect_face are already available, reuse them
        if self._landmarks_ready and self.last_landmarks:
            return self.last_landmarks
        # Otherwise, try to detect face/landmarks now
        face = self.detect_face(frame)
        if face is None:
            return []
        return self.last_landmarks

    def detect_head_pose(self, frame):
        """Estimate head orientation; returns 'left', 'right', 'up', 'down', or 'center'. """
        # Avoid duplicate computation: if detect_face was just called on this frame, use stored landmarks
        if self._landmarks_ready and self.last_landmarks:
            landmarks = self.last_landmarks
        else:
            # Otherwise, run face detection now
            face_box = self.detect_face(frame)
            if face_box is None or not self.last_landmarks:
                # No face found
                return "center"
            landmarks = self.last_landmarks
        # We've used the landmarks for this frame, reset the flag
        self._landmarks_ready = False

        # Need sufficient landmarks to compute pose
        if len(landmarks) < 1:
            return "center"
        # Select key landmark points for pose estimation (MediaPipe FaceMesh indices)
        image_points = np.array([
            landmarks[4],    # Nose tip
            landmarks[152],  # Chin
            landmarks[263],  # Left eye outer corner
            landmarks[130],  # Right eye outer corner
            landmarks[291],  # Left mouth corner
            landmarks[61]    # Right mouth corner
        ], dtype="double")
        # 3D model points corresponding to the above landmarks (an approximate average face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype="double")
        # Camera intrinsic matrix (approximate focal length = frame width)
        h, w = frame.shape[:2]
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w/2],
                                [0, focal_length, h/2],
                                [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))  # assume no lens distortion
        # SolvePnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return "center"
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        # Get Euler angles from the rotation matrix
        proj_mat = np.hstack((rmat, tvec))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)
        pitch = float(eulerAngles[0]); yaw = float(eulerAngles[1]); roll = float(eulerAngles[2])
        # Normalize angles to [-180,180], then constrain to [-90,90] for easier interpretation
        if pitch > 90: pitch -= 180
        if pitch < -90: pitch += 180
        if yaw > 90: yaw -= 180
        if yaw < -90: yaw += 180
        # Store raw angles for debugging/display
        self.last_raw_pitch = pitch
        self.last_raw_yaw = yaw
        # Record history for adaptive thresholding
        self.angle_history.append((pitch, yaw))
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        # Update adaptive thresholds after enough frames
        if self.adaptive_thresholds and len(self.angle_history) > 10:
            # Exclude extreme outliers by taking the 3rd largest/smallest values
            sorted_pitch = sorted(p for p, _ in self.angle_history)
            sorted_yaw = sorted(abs(y) for _, y in self.angle_history)
            # Update observed extremes
            if sorted_pitch[-3] > self.pitch_max: 
                self.pitch_max = sorted_pitch[-3]
            if sorted_pitch[2] < self.pitch_min: 
                self.pitch_min = sorted_pitch[2]
            if sorted_yaw[-3] > self.yaw_max: 
                self.yaw_max = sorted_yaw[-3]
        # Determine pose label
        if self.initial_frames < 15:
            # During initial calibration period, don't trust movements (assume neutral)
            self.initial_frames += 1
            new_pose = "center"
        else:
            # Use adaptive thresholds if enabled and we have sufficient history
            if self.adaptive_thresholds and len(self.angle_history) > 15:
                # Compute adaptive thresholds as a percentage of observed range (with minimum floors)
                right_thresh = max(12.0, 0.7 * self.yaw_max)
                left_thresh  = max(12.0, 0.7 * self.yaw_max)
                down_thresh  = max(8.0,  0.6 * self.pitch_max)
                up_thresh    = max(12.0, 0.6 * abs(self.pitch_min))
            else:
                # Use fixed default thresholds
                right_thresh = self.pose_thresholds["right"]
                left_thresh  = self.pose_thresholds["left"]
                down_thresh  = self.pose_thresholds["down"]
                up_thresh    = self.pose_thresholds["up"]
            # Classify based on yaw and pitch
            if yaw > right_thresh:
                new_pose = "right"
            elif yaw < -left_thresh:
                new_pose = "left"
            elif pitch > down_thresh:
                new_pose = "down"
            elif pitch < -up_thresh:
                new_pose = "up"
            else:
                new_pose = "center"
        # Smooth output: require pose to hold for a couple frames (except returning to center which can be faster)
        if new_pose == self.last_pose:
            self.pose_counter += 1
        else:
            self.pose_counter = 0
        if self.pose_counter >= self.pose_consec_frames or (new_pose == "center" and self.pose_counter >= 1):
            # Commit the new pose after it has been stable
            self.last_pose = new_pose
        return self.last_pose

    def detect_blink(self, frame):
        """Detect blink using Eye Aspect Ratio (EAR) on Mediapipe face landmarks."""
        face_box = self.detect_face(frame)
        landmarks = self.last_landmarks
        # Reset landmark-ready flag
        self._landmarks_ready = False
        if face_box is None or not landmarks:
            self.blink_counter = 0
            return False
        # MediaPipe FaceMesh indices for eye landmarks
        left_outer = landmarks[33]
        left_inner = landmarks[133]
        left_top = landmarks[159]
        left_bottom = landmarks[145]
        right_outer = landmarks[263]
        right_inner = landmarks[362]
        right_top = landmarks[386]
        right_bottom = landmarks[374]
        # Compute distances
        left_horiz = np.linalg.norm(np.array(left_outer) - np.array(left_inner))
        left_vert = np.linalg.norm(np.array(left_top) - np.array(left_bottom))
        right_horiz = np.linalg.norm(np.array(right_outer) - np.array(right_inner))
        right_vert = np.linalg.norm(np.array(right_top) - np.array(right_bottom))
        # Compute EAR for each eye
        left_ear = left_vert / (left_horiz + 1e-6)
        right_ear = right_vert / (right_horiz + 1e-6)
        ear_avg = (left_ear + right_ear) / 2.0
        # Blink detection logic
        blink_detected = False
        if ear_avg < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_consec_frames:
                blink_detected = True
            self.blink_counter = 0
        return blink_detected

    def test_camera(self):
        """Quick test to check if a webcam can be opened."""
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

    def process_frame(self, frame):
        """Process a frame and return a dict with head pose and blink status."""
        pose = self.detect_head_pose(frame)
        blink = self.detect_blink(frame)
        return {"pose": pose, "blink": blink}