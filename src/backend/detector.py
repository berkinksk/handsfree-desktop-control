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
import cv2.data as _cv2_data

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
        # load profile face cascade for side poses
        profile_path = os.path.join(_cv2_data.haarcascades, 'haarcascade_profileface.xml')
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
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

        # blink detection parameters
        self.blink_threshold = 0.2
        self.blink_consec_frames = 3
        self.blink_counter = 0
        # head pose threshold (degrees) for left/right/up/down decisions
        self.pose_threshold = 15.0
        # direction-specific thresholds (degrees)
        self.pose_thresholds = {
            "left": 15.0,
            "right": 15.0, 
            "up": 15.0,
            "down": 10.0  # even lower threshold for downward movement (most challenging to detect)
        }
        # head pose smoothing parameters
        self.last_pose = "center"
        self.pose_consec_frames = 2  # reduced to make more responsive
        self.pose_counter = 0
        # store last raw angles for visualization
        self.last_raw_pitch = 0.0
        self.last_raw_yaw = 0.0

        # adaptive thresholds - these will adjust to user's range of motion
        self.adaptive_thresholds = True  # enable adaptive thresholds by default
        self.pitch_max = 0.0  # will track the max positive pitch
        self.pitch_min = 0.0  # will track the min negative pitch
        self.yaw_max = 0.0    # will track max absolute yaw (either direction)
        self.angle_history = []  # stores recent angle measurements
        self.history_size = 20   # number of frames to keep in history
        self.initial_frames = 0  # count frames for calibration

    def detect_head_pose(self, frame):
        """Estimate head orientation and return one of: left, right, up, down, center."""
        # downscale frame for robust landmark detection
        h, w = frame.shape[:2]
        resize_width = 320
        scale = resize_width / float(w)
        h_s = int(h * scale)
        frame_small = cv2.resize(frame, (resize_width, h_s))
        # detect face and landmarks on small frame
        face_small = self.detect_face(frame_small)
        # if small-frame face detection fails, fallback to full-frame detection
        if face_small is None:
            face_full = self.detect_face(frame)
            if face_full is not None:
                fx, fy, fw, fh = face_full
                # map full-frame box to small-frame coordinates
                face_small = (int(fx * scale), int(fy * scale), int(fw * scale), int(fh * scale))
                print(f"[DEBUG] fallback face: full={face_full} -> small={face_small}")
        landmarks_small = self.detect_landmarks(frame_small, face_small)
        # flatten nested landmarks list if necessary
        flat_small_landmarks = []
        if landmarks_small:
            # if result is nested (one array of points inside a list)
            if isinstance(landmarks_small[0], (list, tuple)) and isinstance(landmarks_small[0][0], (list, tuple)):
                flat_small_landmarks = landmarks_small[0]
            else:
                flat_small_landmarks = landmarks_small
        # debug: report small-frame bbox and flattened landmark count
        print(f"[DEBUG] small detect: face_small={face_small}, landmarks_small_count={len(flat_small_landmarks)}")
        # need at least six points for solvePnP
        if face_small is None or len(flat_small_landmarks) < 6:
            return "center"
        # map landmarks to original resolution
        landmarks = [(pt[0] / scale, pt[1] / scale) for pt in flat_small_landmarks]
        # we no longer use full-frame face box here
        face = None

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
        # Compose projection matrix and decompose to Euler angles
        proj_mat = np.hstack((rmat, tvec))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)
        # extract pitch, yaw, roll (in degrees)
        pitch = float(eulerAngles[0])
        yaw   = float(eulerAngles[1])
        roll  = float(eulerAngles[2])
        # normalize angles to [-90, 90] for interpretation
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180
        # print normalized values
        print(f"[DEBUG] norm_pitch={pitch:.2f}°, norm_yaw={yaw:.2f}°, norm_roll={roll:.2f}°")
        
        # store last raw angles for visualization
        self.last_raw_pitch = pitch
        self.last_raw_yaw = yaw
        
        # update angle history for adaptive thresholds
        self.angle_history.append((pitch, yaw))
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
            
        # update max/min values for adaptive thresholds
        if self.adaptive_thresholds and len(self.angle_history) > 10:
            # update pitch max/min (excluding outliers)
            sorted_pitch = sorted([p for p, _ in self.angle_history])
            if sorted_pitch[-3] > self.pitch_max:  # use 3rd highest to avoid outliers
                self.pitch_max = sorted_pitch[-3]
            if sorted_pitch[2] < self.pitch_min:  # use 3rd lowest to avoid outliers
                self.pitch_min = sorted_pitch[2]
                
            # update max absolute yaw (excluding outliers)
            abs_yaws = sorted([abs(y) for _, y in self.angle_history])
            if abs_yaws[-3] > self.yaw_max:  # use 3rd highest to avoid outliers
                self.yaw_max = abs_yaws[-3]
        
        # first 30 frames used for calibration - just return center
        if self.initial_frames < 30:
            self.initial_frames += 1
            new_pose = "center"
        # after calibration, use either fixed or adaptive thresholds
        else:
            # use adaptive thresholds if enabled and enough history is available
            if self.adaptive_thresholds and len(self.angle_history) > 15:
                # Use a percentage of the observed range for thresholds
                right_thresh = max(12.0, 0.7 * self.yaw_max)
                left_thresh = max(12.0, 0.7 * self.yaw_max)
                down_thresh = max(8.0, 0.6 * self.pitch_max)
                up_thresh = max(12.0, 0.6 * abs(self.pitch_min))
                
                # Log the current adaptive thresholds occasionally (every 30 frames)
                if (self.initial_frames % 30) == 0:
                    print(f"[ADAPTIVE] Thresholds: R={right_thresh:.1f}° L={left_thresh:.1f}° D={down_thresh:.1f}° U={up_thresh:.1f}°")
                    print(f"[ADAPTIVE] Range: yaw_max={self.yaw_max:.1f}° pitch_max={self.pitch_max:.1f}° pitch_min={self.pitch_min:.1f}°")
            else:
                # Use fixed thresholds
                right_thresh = self.pose_thresholds["right"]
                left_thresh = self.pose_thresholds["left"]
                down_thresh = self.pose_thresholds["down"] 
                up_thresh = self.pose_thresholds["up"]
                
            # Classify pose based on current pitch/yaw and thresholds
            if yaw > right_thresh:
                new_pose = "right"
            elif yaw < -left_thresh:
                new_pose = "left"
            elif pitch > down_thresh:  # looking down (positive pitch)
                new_pose = "down"
            elif pitch < -up_thresh:  # looking up (negative pitch)
                new_pose = "up"
            else:
                new_pose = "center"
        
        # Apply smoothing for stable output
        if new_pose == self.last_pose:
            self.pose_counter += 1
        else:
            self.pose_counter = 0
            
        # Return smoothed pose, only change after consecutive detections
        if self.pose_counter >= self.pose_consec_frames:
            self.last_pose = new_pose
        # Special case for center pose - don't require as many frames to go back to center
        elif new_pose == "center" and self.pose_counter >= 1:
            self.last_pose = new_pose
            
        return self.last_pose

    def detect_blink(self, frame):
        """Detect blinks using Eye Aspect Ratio (EAR) method."""
        # detect face and landmarks
        face = self.detect_face(frame)
        landmarks = self.detect_landmarks(frame, face)
        if face is None or not landmarks or len(landmarks) < 68:
            # reset blink counter if no face or insufficient landmarks
            self.blink_counter = 0
            return False

        # compute Eye Aspect Ratio (EAR)
        def compute_ear(eye):
            p1, p2, p3, p4, p5, p6 = eye
            vert1 = np.linalg.norm(np.array(p2) - np.array(p6))
            vert2 = np.linalg.norm(np.array(p3) - np.array(p5))
            horiz = np.linalg.norm(np.array(p1) - np.array(p4))
            return (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0.0

        # extract eye landmarks
        left_eye = [landmarks[i] for i in range(36, 42)]
        right_eye = [landmarks[i] for i in range(42, 48)]
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        ear_avg = (left_ear + right_ear) / 2.0

        blink_detected = False
        # update counter based on EAR threshold
        if ear_avg < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_consec_frames:
                blink_detected = True
            # reset counter after eye opens
            self.blink_counter = 0
        return blink_detected

    def detect_face(self, frame):
        """Detect a face in the frame and return its bounding box (x, y, w, h) or None."""
        # if cascade failed to load, skip detection
        if self.face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            return faces[0]
        # fallback to profile cascade (frontal side view)
        if not self.profile_cascade.empty():
            faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                return faces[0]
            # try flipped image for opposite profile
            gray_flipped = cv2.flip(gray, 1)
            faces = self.profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # map x from flipped to original coordinates
                x = frame.shape[1] - x - w
                return (x, y, w, h)
        return None

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
        # landmarks[0] may be shape (1,N,2) or (N,2)
        pts = landmarks[0]
        # if there's an extra first dimension, remove it
        if isinstance(pts, np.ndarray) and pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        # now pts is expected to be (N,2)
        return pts.tolist()

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

    def process_frame(self, frame):
        """Process a frame and return head orientation and blink status."""
        # get head orientation label
        pose = self.detect_head_pose(frame)
        # get blink detection status
        blink = self.detect_blink(frame)
        # return structured result
        return {'pose': pose, 'blink': blink} 