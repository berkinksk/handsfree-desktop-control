import cv2
import numpy as np
import mediapipe as mp

class HeadEyeDetector:
    """Detect head pose (orientation) and blinks from video frames using MediaPipe Face Mesh."""
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 blink_threshold=0.27,
                 blink_consec_frames=3):
        # Initialize MediaPipe Face Mesh solution
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # set True to get iris landmarks if needed for precision
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Blink detection parameters
        self.blink_threshold = blink_threshold    # EAR threshold for eyes considered closed
        self.blink_consec_frames = blink_consec_frames  # how many consecutive frames below threshold to count a blink
        self.blink_counter = 0    # counter for consecutive frames of eyes below threshold
        self.total_blinks = 0     # total blinks detected
        # Predefine indices for landmarks used in pose estimation (2D/3D points)
        # Using tip of nose, outer eye corners, mouth corners, and chin point
        self.pose_landmark_indices = [1, 33, 263, 61, 291, 199]
        
        # Store last computed angles (will store smoothed angles)
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0

        # Smoothing parameters for pose angles
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        self.smoothing_alpha = 0.4 # Smoothing factor (0 < alpha <= 1). Smaller = more smoothing.

        self.face_landmarks_for_drawing = None # Store landmarks for drawing

    def _calculate_head_pose(self, landmarks, frame_shape):
        """
        Internal method to calculate head pose from processed landmarks.
        Updates self.last_pitch, self.last_yaw, self.last_roll.
        Returns one of {"left", "right", "up", "down", "center"} indicating head orientation.
        """
        img_h, img_w, _ = frame_shape
        face2d_pts = []
        face3d_pts = []
        for idx in self.pose_landmark_indices:
            lm = landmarks[idx]
            # Convert normalized landmark coordinates to pixel coordinates
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face2d_pts.append([x, y])
            # Construct the 3D point. 
            # Using a more constant scaling for Z to improve stability.
            # The exact scale of Z matters less than its relative values for solvePnP with a calibrated camera,
            # but for an estimated camera, extreme Z values can hurt.
            z_scale_factor = 60.0 # An arbitrary scale factor for depth (reverted from 120.0)
            if idx == 1: # Nose landmark
                face3d_pts.append([x, y, lm.z * z_scale_factor * 1.5]) # Keep nose slightly more prominent
            else:
                face3d_pts.append([x, y, lm.z * z_scale_factor])

        face2d_pts = np.array(face2d_pts, dtype=np.float64)
        face3d_pts = np.array(face3d_pts, dtype=np.float64)

        # Define camera matrix (assuming no lens distortion and focal length ~ frame width)
        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no distortion

        # SolvePnP to get rotation vector
        success, rot_vec, trans_vec = cv2.solvePnP(face3d_pts, face2d_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            # solvePnP failed
            self.last_pitch, self.last_yaw, self.last_roll = 0.0, 0.0, 0.0
            return "center"

        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Calculate raw Euler angles based on the rotation matrix
        sy = np.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
        singular = sy < 1e-6

        raw_physical_pitch: float
        raw_physical_yaw: float
        raw_physical_roll: float

        if not singular:
            # Physical Pitch (nod up/down, rotation around X-axis from camera perspective)
            raw_physical_pitch = np.arctan2(rot_mat[2,1], rot_mat[2,2]) * 180 / np.pi
            # Physical Yaw (turn left/right, rotation around Y-axis from camera perspective)
            raw_physical_yaw   = np.arctan2(-rot_mat[2,0], sy) * 180 / np.pi
            # Physical Roll (tilt side-to-side, rotation around Z-axis from camera perspective)
            raw_physical_roll  = np.arctan2(rot_mat[1,0], rot_mat[0,0]) * 180 / np.pi
        else: # Gimbal lock case
            raw_physical_yaw   = np.arctan2(-rot_mat[2,0], sy) * 180 / np.pi 
            raw_physical_pitch = 0.0 # Conventionally set to 0 in this gimbal lock case
            raw_physical_roll  = np.arctan2(-rot_mat[0,2], rot_mat[1,1]) * 180 / np.pi # Derived
        
        # Apply EMA smoothing to the correctly mapped physical angles
        self.smooth_pitch = self.smoothing_alpha * raw_physical_pitch + (1 - self.smoothing_alpha) * self.smooth_pitch
        self.smooth_yaw   = self.smoothing_alpha * raw_physical_yaw   + (1 - self.smoothing_alpha) * self.smooth_yaw
        self.smooth_roll  = self.smoothing_alpha * raw_physical_roll  + (1 - self.smoothing_alpha) * self.smooth_roll

        # Use smoothed values for pose logic
        pitch_for_logic = self.smooth_pitch
        yaw_for_logic   = self.smooth_yaw

        # Store smoothed angles for returning as main angle values (previously named raw_... in return dict)
        self.last_pitch = pitch_for_logic
        self.last_yaw   = yaw_for_logic
        self.last_roll  = self.smooth_roll # self.smooth_roll is used directly for consistency here

        # Determine head orientation based on pitch/yaw thresholds
        pose_label = "center" # Default to center

        # Thresholds for determining pose
        yaw_threshold_strong = 6  # Degrees for definite left/right
        pitch_threshold_strong = 12 # Degrees for definite up/down
        center_threshold_yaw = 3 # Degrees for yaw to be considered centered
        center_threshold_pitch = 7 # Degrees for pitch to be considered centered

        is_centered_yaw = abs(yaw_for_logic) < center_threshold_yaw
        is_centered_pitch = abs(pitch_for_logic) < center_threshold_pitch

        if is_centered_yaw and is_centered_pitch:
            pose_label = "center"
        elif yaw_for_logic > yaw_threshold_strong:
            pose_label = "right"
        elif yaw_for_logic < -yaw_threshold_strong:
            pose_label = "left"
        elif pitch_for_logic > pitch_threshold_strong:
            pose_label = "up"
        elif pitch_for_logic < -pitch_threshold_strong:
            pose_label = "down"
        # If conditions are between center_threshold and strong_threshold, 
        # it will remain "center" unless a strong threshold is met.
        # This creates a hysteresis effect, reducing rapid changes for small movements around the strong thresholds.
        # For more complex scenarios (e.g., diagonal movements like "up-right"),
        # additional logic would be needed here.

        return pose_label

    def _calculate_blink(self, landmarks, frame_shape):
        """
        Internal method to calculate blink status from processed landmarks.
        Updates self.blink_counter and self.total_blinks.
        Returns True if a blink is detected on this frame (i.e., a full blink event completed).
        """
        img_h, img_w, _ = frame_shape
        # Define indices around the eyes for EAR calculation (using Mediapipe landmarks)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [263, 387, 385, 362, 380, 373]
        
        left_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in left_eye_indices]
        right_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in right_eye_indices]

        def eye_aspect_ratio(eye_points):
            p1, p2, p3, p4, p5, p6 = eye_points
            vert1 = np.linalg.norm(np.array(p2) - np.array(p6))
            vert2 = np.linalg.norm(np.array(p3) - np.array(p5))
            horiz = np.linalg.norm(np.array(p1) - np.array(p4))
            return (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0

        left_EAR = eye_aspect_ratio(left_eye_pts)
        right_EAR = eye_aspect_ratio(right_eye_pts)
        ear_avg = (left_EAR + right_EAR) / 2.0

        blink_detected_this_frame = False
        if ear_avg < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_consec_frames:
                self.total_blinks += 1
                blink_detected_this_frame = True
            self.blink_counter = 0
        
        return blink_detected_this_frame, ear_avg

    def process_frame_and_detect_features(self, frame):
        """
        Processes a single video frame to detect head pose and blinks.
        This is the main method to call for each frame.
        Returns a dictionary with detection results.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        self.face_landmarks_for_drawing = None # Reset landmarks for drawing

        if results.multi_face_landmarks:
            # Use the first face (since max_num_faces=1)
            face_landmarks_object = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks_object.landmark # The list of landmark objects
            self.face_landmarks_for_drawing = face_landmarks_object


            pose_label = self._calculate_head_pose(landmarks_list, frame.shape)
            blink_detected, ear_avg = self._calculate_blink(landmarks_list, frame.shape)
            
            return {
                "pose": pose_label,
                "blink_detected": blink_detected,
                "landmarks_mp_object": self.face_landmarks_for_drawing, # For mp_drawing
                "raw_pitch": self.last_pitch,
                "raw_yaw": self.last_yaw,
                "raw_roll": self.last_roll,
                "total_blinks": self.total_blinks,
                "ear_avg": ear_avg
            }
        else:
            # No face detected
            self.last_pitch, self.last_yaw, self.last_roll = 0.0, 0.0, 0.0
            self.blink_counter = 0 # Reset blink counter if no face
            return {
                "pose": "center",
                "blink_detected": False,
                "landmarks_mp_object": None,
                "raw_pitch": 0.0,
                "raw_yaw": 0.0,
                "raw_roll": 0.0,
                "total_blinks": self.total_blinks,
                "ear_avg": 0.0
            }

# Removed old detect_head_pose and detect_blink methods as their logic is now in
# _calculate_head_pose, _calculate_blink, and process_frame_and_detect_features.
# The original methods are essentially replaced by process_frame_and_detect_features.

# Example usage (for illustration, not part of the class):
# if __name__ == '__main__':
#     detector = HeadEyeDetector()
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         detection_results = detector.process_frame_and_detect_features(frame)
#         print(f"Pose: {detection_results['pose']}, Blink: {detection_results['blink_detected']}, Total Blinks: {detection_results['total_blinks']}")
#         # Further processing like drawing landmarks using detection_results['landmarks_mp_object']
#         # cv2.imshow('Frame', frame) # Assuming drawing happens on the frame
#         if cv2.waitKey(5) & 0xFF == 27: # ESC key
#             break
#     cap.release()
#     cv2.destroyAllWindows()
