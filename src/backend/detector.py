import cv2
import numpy as np
import mediapipe as mp

class HeadEyeDetector:
    """Detect head pose (orientation) and pupil-based actions (with hold) from video frames using MediaPipe Face Mesh."""
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 norm_pupil_y_diff_threshold_center=0.040, # For 'center' pose (was 0.080)
                 norm_pupil_y_diff_threshold_up=0.020,     # For 'up' pose (was 0.030)
                 norm_pupil_y_diff_threshold_down=0.075,   # For 'down' pose
                 norm_pupil_y_diff_threshold_left=0.025,   # For 'left' pose (was 0.035)
                 norm_pupil_y_diff_threshold_right=0.060,  # For 'right' pose
                 pupil_action_consec_frames=23, # Approx. 0.75 sec at 30 FPS for held action
                 **kwargs):
        # Initialize MediaPipe Face Mesh solution
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Pupil landmarks require refine_landmarks=True
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Pupil action parameters
        # These thresholds determine sensitivity to the normalized vertical difference between pupil landmarks.
        # A larger Y difference (e.g., one pupil higher than the other, potentially due to eyelid movement closing over one pupil more)
        # normalized by the horizontal distance between pupils, can trigger an "action".
        # This is currently used as a proxy for blink detection.
        # Tuning these values requires empirical testing based on user, lighting, and desired responsiveness.
        # - Higher values make action detection less sensitive.
        # - Lower values make action detection more sensitive.
        # Consider that head tilt (roll) might also affect this metric if not perfectly compensated.
        self.norm_pupil_y_diff_thresholds = {
            "center": norm_pupil_y_diff_threshold_center, # Threshold when head is centered
            "up": norm_pupil_y_diff_threshold_up,         # Threshold when head is tilted up
            "down": norm_pupil_y_diff_threshold_down,     # Threshold when head is tilted down
            "left": norm_pupil_y_diff_threshold_left,     # Threshold when head is turned left
            "right": norm_pupil_y_diff_threshold_right    # Threshold when head is turned right
        }
        # Number of consecutive frames the pupil y-difference must exceed the threshold to trigger a held action.
        # E.g., 23 frames at ~30 FPS is roughly 0.75 seconds.
        # Adjust for desired hold duration before an action is registered.
        self.pupil_action_consec_frames = pupil_action_consec_frames
        self.pupil_action_hold_counter = 0 # Counter for frames norm_pupil_y_diff is above threshold
        self.total_actions = 0     # Total actions detected
        self.action_pending_reset = False # True if action triggered and waiting for condition to cease

        # Predefine indices for landmarks used in pose estimation
        self.pose_landmark_indices = [1, 33, 263, 61, 291, 199]
 
        # Predefine indices for pupil landmarks (approximate center of pupils)
        # Left pupil: 473, Right pupil: 468
        self.left_pupil_idx = 473
        self.right_pupil_idx = 468
        
        # Store last computed angles (will store smoothed angles)
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0

        # Smoothing parameters for pose angles
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        self.smoothing_alpha = 0.4

        self.face_landmarks_for_drawing = None 
        self.last_raw_physical_yaw = 0.0 
        self.last_raw_physical_pitch = 0.0 
        self.last_raw_physical_roll = 0.0

        # Thresholds for pose determination (degrees)
        self.yaw_threshold_strong = float(kwargs.get('yaw_threshold_strong', 5.0))
        self.yaw_threshold_strong_left = float(kwargs.get('yaw_threshold_strong_left', 5.0))
        self.pitch_threshold_up = float(kwargs.get('pitch_threshold_up', 1.7))
        self.pitch_threshold_down = float(kwargs.get('pitch_threshold_down', 0.7)) # User changed from 0.9

        # New: Tolerances for 2D pose zones
        self.pitch_tolerance_for_lr_pose = float(kwargs.get('pitch_tolerance_for_lr_pose', 2.0))
        self.yaw_tolerance_for_ud_pose = float(kwargs.get('yaw_tolerance_for_ud_pose', 2.0))

        self.roll_threshold_center = float(kwargs.get('roll_threshold_center', 5.0))

        # Hysteresis parameters (from previous implementation, ensure they are present)
        self.committed_pose_label = "center" # For hysteresis
        self.hysteresis_yaw_neutral_threshold = float(kwargs.get('hysteresis_yaw_neutral_threshold', 1.5)) # Needs to pass this towards center to change from L/R
        self.hysteresis_pitch_neutral_threshold = float(kwargs.get('hysteresis_pitch_neutral_threshold', 0.5)) # Needs to pass this towards center to change from U/D

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
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face2d_pts.append([x, y])
            z_scale_factor = 60.0 
            if idx == 1: 
                face3d_pts.append([x, y, lm.z * z_scale_factor * 1.5]) 
            else:
                face3d_pts.append([x, y, lm.z * z_scale_factor])

        face2d_pts = np.array(face2d_pts, dtype=np.float64)
        face3d_pts = np.array(face3d_pts, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face3d_pts, face2d_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            self.last_pitch, self.last_yaw, self.last_roll = 0.0, 0.0, 0.0
            return "center"

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        sy = np.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
        singular = sy < 1e-6

        raw_physical_pitch: float
        raw_physical_yaw: float
        raw_physical_roll: float

        if not singular:
            raw_physical_pitch = np.arctan2(rot_mat[2,1], rot_mat[2,2]) * 180 / np.pi
            raw_physical_yaw   = np.arctan2(-rot_mat[2,0], sy) * 180 / np.pi
            raw_physical_roll  = np.arctan2(rot_mat[1,0], rot_mat[0,0]) * 180 / np.pi
        else: 
            raw_physical_yaw   = np.arctan2(-rot_mat[2,0], sy) * 180 / np.pi 
            raw_physical_pitch = 0.0 
            raw_physical_roll  = np.arctan2(-rot_mat[0,2], rot_mat[1,1]) * 180 / np.pi
        
        self.last_raw_physical_yaw = raw_physical_yaw
        self.last_raw_physical_pitch = raw_physical_pitch
        self.last_raw_physical_roll = raw_physical_roll

        self.smooth_pitch = self.smoothing_alpha * raw_physical_pitch + (1 - self.smoothing_alpha) * self.smooth_pitch
        self.smooth_yaw   = self.smoothing_alpha * raw_physical_yaw   + (1 - self.smoothing_alpha) * self.smooth_yaw
        self.smooth_roll  = self.smoothing_alpha * raw_physical_roll  + (1 - self.smoothing_alpha) * self.smooth_roll

        pitch_for_logic = self.smooth_pitch
        yaw_for_logic   = self.smooth_yaw

        self.last_pitch = pitch_for_logic
        self.last_yaw   = yaw_for_logic
        self.last_roll  = self.smooth_roll

        # NOTE on Roll: While roll is calculated and smoothed, it's not directly used in the current
        # discrete pose_label logic (center, left, right, up, down).
        # However, its stability can affect the overall PnP solution and perceived smoothness.
        # Monitor its behavior (e.g., self.last_raw_physical_roll, self.last_roll) during testing,
        # especially if pitch/yaw seem unstable or if head tilt significantly impacts controls.
        # pose_label = "center" # Old simple default

        # Determine the 'raw' pose for this frame based on current thresholds and new 2D zone logic
        calculated_pose_label = "center" # Default for this frame's calculation

        # Check for Left/Right pose first, as yaw often has larger magnitude changes
        if yaw_for_logic <= -self.yaw_threshold_strong_left and abs(pitch_for_logic) < self.pitch_tolerance_for_lr_pose:
            calculated_pose_label = "left"
        elif yaw_for_logic > self.yaw_threshold_strong and abs(pitch_for_logic) < self.pitch_tolerance_for_lr_pose:
            calculated_pose_label = "right"
        # Then check for Up/Down pose if not already Left/Right
        elif pitch_for_logic > self.pitch_threshold_up and abs(yaw_for_logic) < self.yaw_tolerance_for_ud_pose:
            calculated_pose_label = "up"
        elif pitch_for_logic < -self.pitch_threshold_down and abs(yaw_for_logic) < self.yaw_tolerance_for_ud_pose:
            calculated_pose_label = "down"
        # If none of the above, it remains "center"

        # Apply Hysteresis (copied and adapted from previous version, ensure it's correctly integrated)
        # If previously committed to "left"
        if self.committed_pose_label == "left":
            # If yaw is still significantly left (more negative than neutral OR within the 'left' zone criteria)
            if yaw_for_logic <= -self.hysteresis_yaw_neutral_threshold or \
               (yaw_for_logic <= -self.yaw_threshold_strong_left and abs(pitch_for_logic) < self.pitch_tolerance_for_lr_pose):
                self.committed_pose_label = "left"
            else:
                self.committed_pose_label = calculated_pose_label # Re-evaluate based on current full logic

        # If previously committed to "right"
        elif self.committed_pose_label == "right":
            if yaw_for_logic >= self.hysteresis_yaw_neutral_threshold or \
               (yaw_for_logic > self.yaw_threshold_strong and abs(pitch_for_logic) < self.pitch_tolerance_for_lr_pose):
                self.committed_pose_label = "right"
            else:
                self.committed_pose_label = calculated_pose_label

        # If previously committed to "up"
        elif self.committed_pose_label == "up":
            if pitch_for_logic >= self.hysteresis_pitch_neutral_threshold or \
               (pitch_for_logic > self.pitch_threshold_up and abs(yaw_for_logic) < self.yaw_tolerance_for_ud_pose):
                self.committed_pose_label = "up"
            else:
                self.committed_pose_label = calculated_pose_label

        # If previously committed to "down"
        elif self.committed_pose_label == "down":
            if pitch_for_logic <= -self.hysteresis_pitch_neutral_threshold or \
               (pitch_for_logic < -self.pitch_threshold_down and abs(yaw_for_logic) < self.yaw_tolerance_for_ud_pose):
                self.committed_pose_label = "down"
            else:
                self.committed_pose_label = calculated_pose_label
        
        # If previously committed to "center" or any other case (e.g. after reset from L/R/U/D)
        else: # This includes self.committed_pose_label == "center"
            self.committed_pose_label = calculated_pose_label # Simply adopt the newly calculated label
        
        return self.committed_pose_label # Return the possibly-hysterisis-modified pose

    def _calculate_pupil_y_diff_action(self, landmarks, frame_shape, pose_label):
        """
        Internal method to calculate pupil y-difference action status based on current pose.
        An action is triggered if NormPupilYDiff > pose_specific_threshold for pupil_action_consec_frames.
        Updates self.pupil_action_hold_counter and self.total_actions.
        Returns True if an action is detected on this frame and the current norm_pupil_y_diff.
        """
        img_h, img_w, _ = frame_shape

        left_pupil_lm = landmarks[self.left_pupil_idx]
        right_pupil_lm = landmarks[self.right_pupil_idx]

        left_pupil_y = left_pupil_lm.y * img_h
        right_pupil_y = right_pupil_lm.y * img_h
        left_pupil_x = left_pupil_lm.x * img_w
        right_pupil_x = right_pupil_lm.x * img_w
        
        pupil_y_pixel_diff = abs(left_pupil_y - right_pupil_y)
        pupil_x_pixel_dist = abs(left_pupil_x - right_pupil_x)

        epsilon = 1e-6 # To prevent division by zero
        normalized_pupil_y_diff = 0.0
        if pupil_x_pixel_dist > epsilon:
            normalized_pupil_y_diff = pupil_y_pixel_diff / pupil_x_pixel_dist
        else: # Unlikely case, but handle it (e.g. if pupils are vertically aligned)
            normalized_pupil_y_diff = pupil_y_pixel_diff / epsilon # Make it a large number if Y diff is non-zero

        action_detected_this_frame = False # Initialize

        # Only proceed with action detection if pose is "center"
        if pose_label != "center":
            self.pupil_action_hold_counter = 0
            self.action_pending_reset = False
            # Return immediately, indicating no action detected for non-center poses
            # and the current (but now irrelevant for action) normalized_pupil_y_diff
            return False, normalized_pupil_y_diff

        # Get the threshold specific to the current pose (which must be "center" at this point)
        current_threshold = self.norm_pupil_y_diff_thresholds["center"]

        # Action detection logic:
        # If the normalized_pupil_y_diff exceeds the current_threshold for the current pose:
        # 1. If an action is not already 'pending_reset' (i.e., we are not in the cooldown phase of a previous action):
        #    a. Increment the hold_counter.
        #    b. If hold_counter reaches pupil_action_consec_frames:
        #       i. An action is registered (total_actions incremented, action_detected_this_frame = True).
        #      ii. Set action_pending_reset = True (enter cooldown; wait for condition to cease).
        #     iii. Reset hold_counter.
        # 2. If an action IS 'pending_reset', keep hold_counter at 0 (don't re-trigger while waiting for reset).
        # If the normalized_pupil_y_diff is below threshold:
        # 1. Reset hold_counter.
        # 2. Set action_pending_reset = False (ready for a new action sequence).
        if normalized_pupil_y_diff > current_threshold:
            if not self.action_pending_reset:
                self.pupil_action_hold_counter += 1
                if self.pupil_action_hold_counter == self.pupil_action_consec_frames:
                    self.total_actions += 1
                    action_detected_this_frame = True
                    self.action_pending_reset = True # Action triggered, now wait for condition to cease
                    self.pupil_action_hold_counter = 0 # Reset counter
            else:
                # Condition still met, but an action was already triggered and is pending reset.
                # Keep counter at 0 to prevent re-counting while in this state.
                self.pupil_action_hold_counter = 0
        else:
            # Normalized PupilYDiff is below threshold, so reset everything for the next action cycle
            self.pupil_action_hold_counter = 0 
            self.action_pending_reset = False # Condition ceased, ready for a new action sequence
            
        return action_detected_this_frame, normalized_pupil_y_diff

    def process_frame_and_detect_features(self, frame):
        """
        Processes a single video frame to detect head pose and pupil actions (with hold).
        Returns a dictionary with detection results.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        self.face_landmarks_for_drawing = None

        if results.multi_face_landmarks and len(results.multi_face_landmarks[0].landmark) > max(self.left_pupil_idx, self.right_pupil_idx):
            face_landmarks_object = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks_object.landmark 
            self.face_landmarks_for_drawing = face_landmarks_object

            pose_label = self._calculate_head_pose(landmarks_list, frame.shape)
            
            action_detected, norm_pupil_y_diff = self._calculate_pupil_y_diff_action(landmarks_list, frame.shape, pose_label)
            
            return {
                "pose": pose_label,
                "action_detected": action_detected,
                "norm_pupil_y_diff": norm_pupil_y_diff,
                "landmarks_mp_object": self.face_landmarks_for_drawing,
                "raw_pitch": self.last_pitch,
                "raw_yaw": self.last_yaw,
                "raw_physical_yaw": self.last_raw_physical_yaw,
                "raw_physical_pitch": self.last_raw_physical_pitch,
                "raw_physical_roll": self.last_raw_physical_roll,
                "raw_roll": self.last_roll,
                "total_actions": self.total_actions,
            }
        else:
            self.last_pitch, self.last_yaw, self.last_roll = 0.0, 0.0, 0.0
            self.pupil_action_hold_counter = 0 # Reset counter if no face
            return {
                "pose": "center",
                "action_detected": False,
                "norm_pupil_y_diff": 0.0,
                "landmarks_mp_object": None,
                "raw_pitch": 0.0,
                "raw_yaw": 0.0,
                "raw_physical_yaw": 0.0,
                "raw_physical_pitch": 0.0,
                "raw_physical_roll": 0.0,
                "raw_roll": 0.0,
                "total_actions": self.total_actions,
            }

# Example usage (for illustration, not part of the class):
# if __name__ == '__main__':
#     detector = HeadEyeDetector(blink_threshold=0.25, blink_consec_frames=30) # Example instantiation
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         detection_results = detector.process_frame_and_detect_features(frame)
#         print(f"Pose: {detection_results['pose']}, Blink: {detection_results['blink_detected']}, EAR: {detection_results['ear_avg']:.2f}, Total Blinks: {detection_results['total_blinks']}")
#         # Further processing like drawing landmarks using detection_results['landmarks_mp_object']
#         # cv2.imshow('Frame', frame) # Assuming drawing happens on the frame
#         if cv2.waitKey(5) & 0xFF == 27: # ESC key
#             break
#     cap.release()
#     cv2.destroyAllWindows()
