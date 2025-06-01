import cv2
import numpy as np
import mediapipe as mp
import time

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
                 pupil_action_consec_frames=10, # Approx. 0.34 sec at 30 FPS for held action (was 15)
                 double_click_window_ms=1500,    # Max time in ms between blinks for a double click (was 1000)
                 min_time_after_action_ms=300, # Min time in ms before another action can start after one completes
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

        # Store timing parameters from init
        self.double_click_window_ms = double_click_window_ms
        self.min_time_after_action_ms = min_time_after_action_ms

        # New state variables for double-click logic
        self.last_blink_time = 0 # Timestamp of the last registered blink event, in ns
        self.first_blink_of_potential_double_click_time = 0 # Timestamp of the first blink if we are waiting for a second, in ns
        # These ..._frames attributes are no longer the primary source for ns conversion in _calculate_pupil_y_diff_action
        # but might be kept if any other logic relies on them or for very rough estimates.
        # For precision, direct conversion from _ms to _ns is used in the action calculation.
        self.double_click_window_frames = int((self.double_click_window_ms / 1000.0) * 30) 
        self.min_frames_after_action = int((self.min_time_after_action_ms / 1000.0) * 30) 
        # self.frames_since_last_action is effectively replaced by checking (current_time_ns - self.last_blink_time)
        # self.frames_since_last_action = self.min_frames_after_action + 1 
        
        self.CLICK_TYPE_NONE = "NO_ACTION"
        self.CLICK_TYPE_SINGLE = "SINGLE_CLICK"
        self.CLICK_TYPE_DOUBLE = "DOUBLE_CLICK"
        
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
        Returns a click type (NO_ACTION, SINGLE_CLICK, DOUBLE_CLICK) and the current norm_pupil_y_diff.
        """
        img_h, img_w, _ = frame_shape
        # Using time.monotonic_ns() for more precise time comparisons directly
        current_time_ns = time.monotonic_ns()

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
            normalized_pupil_y_diff = pupil_y_pixel_diff / epsilon

        action_type_this_frame = self.CLICK_TYPE_NONE
        
        # Increment frames_since_last_action if it's not already maxed out to prevent overflow
        # This state is now managed by self.last_action_finish_time_ns
        # self.frames_since_last_action +=1 

        if pose_label != "center":
            self.pupil_action_hold_counter = 0
            self.action_pending_reset = False 
            # If pose is not center, any pending first blink for a double click should be timed out / canceled.
            if self.first_blink_of_potential_double_click_time > 0:
                 # This pending blink is effectively cancelled, not treated as single.
                 self.first_blink_of_potential_double_click_time = 0
            return self.CLICK_TYPE_NONE, normalized_pupil_y_diff

        current_threshold = self.norm_pupil_y_diff_thresholds["center"]
        
        # Convert ms thresholds to ns for direct comparison
        double_click_window_ns = int(self.double_click_window_ms * 1_000_000)
        min_time_after_action_ns = int(self.min_time_after_action_ms * 1_000_000)


        # Check if enough time has passed since the last action completed
        can_initiate_new_action = (current_time_ns - self.last_blink_time) > min_time_after_action_ns

        # Timeout check for a pending first blink (to confirm it as a single click)
        # This must happen BEFORE processing a new blink, in case the new blink is what causes the timeout.
        if self.first_blink_of_potential_double_click_time > 0 and \
           (current_time_ns - self.first_blink_of_potential_double_click_time > double_click_window_ns):
            # First blink timed out, register it as a single click
            # Ensure we don't override a double click that might have just been formed by the current blink
            if action_type_this_frame == self.CLICK_TYPE_NONE: # Only if no action decided for this frame yet
                action_type_this_frame = self.CLICK_TYPE_SINGLE
                self.total_actions += 1
                self.last_blink_time = self.first_blink_of_potential_double_click_time # The time of the actual blink event
            self.first_blink_of_potential_double_click_time = 0 # Reset regardless

        # Now, process current frame's pupil state
        if normalized_pupil_y_diff > current_threshold:
            if can_initiate_new_action and not self.action_pending_reset:
                self.pupil_action_hold_counter += 1
                if self.pupil_action_hold_counter >= self.pupil_action_consec_frames:
                    # Valid "physical" blink detected (held for enough frames)
                    self.pupil_action_hold_counter = 0 # Reset for next detection cycle
                    self.action_pending_reset = True # A physical blink completed, wait for y_diff to go below threshold
                                                     # before another physical blink can start accumulating.

                    # This physical blink is a candidate for click logic
                    current_physical_blink_time = current_time_ns 

                    if self.first_blink_of_potential_double_click_time > 0:
                        # A first blink was pending. Is this current one the second for a double?
                        time_diff_ns = current_physical_blink_time - self.first_blink_of_potential_double_click_time
                        
                        if time_diff_ns <= double_click_window_ns:
                            # Yes, it's a double click!
                            if action_type_this_frame == self.CLICK_TYPE_SINGLE:
                                # This implies the timeout logic above just fired for the first blink.
                                # This is a rare edge case: timeout just made it single, but a new blink arrived
                                # almost simultaneously making it double. Double overrides.
                                self.total_actions -= 1 # Correct the single click count from timeout
                                
                            action_type_this_frame = self.CLICK_TYPE_DOUBLE
                            self.total_actions += 1 
                            self.last_blink_time = current_physical_blink_time # Time of the second blink finishing the double
                            self.first_blink_of_potential_double_click_time = 0 # Reset double click tracking
                        else:
                            # No, current physical blink is too late for a double with the previous one.
                            # The previous first_blink_of_potential_double_click_time has *already* been processed by timeout logic
                            # (or will be in the next frame if it just timed out this exact frame before this check).
                            # So, this current physical blink starts a NEW potential double click.
                            # If the timeout for the previous one hasn't resulted in action_type_this_frame yet,
                            # it will on the next frame, or earlier in this frame if current_time_ns was just past its window.
                            # No click action for *this* frame from *this* physical blink yet.
                            self.first_blink_of_potential_double_click_time = current_physical_blink_time
                    else:
                        # No first blink pending, so this current physical blink is the first of a new potential double.
                        # Or, it could be a single click if no second one follows.
                        # Don't signal action yet. Set it as pending.
                        self.first_blink_of_potential_double_click_time = current_physical_blink_time
                        # Ensure last_blink_time is not updated here, only on confirmed actions.
            # else: (if not can_initiate_new_action or self.action_pending_reset is true)
                # In cooldown from a previous action, or y_diff still high from current blink.
                # pupil_action_hold_counter remains 0 or isn't incremented further.
                pass

        else: # normalized_pupil_y_diff <= current_threshold (physical blink ended or not started)
            self.pupil_action_hold_counter = 0 # Reset physical blink hold counter
            self.action_pending_reset = False  # Ready for a new physical blink to start accumulating
            # If a first_blink was pending and now y_diff is low, the timeout logic will handle it.
            # No direct action here.

        # If a timeout occurred earlier and set action_type_this_frame, it will be returned.
        # If a double click occurred, it will be returned.
        # Otherwise, CLICK_TYPE_NONE is returned.
            
        return action_type_this_frame, normalized_pupil_y_diff

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
            
            # Changed from action_detected to action_type
            action_type, norm_pupil_y_diff = self._calculate_pupil_y_diff_action(landmarks_list, frame.shape, pose_label)
            
            return {
                "pose": pose_label,
                "action_type": action_type, # Changed key name
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
                "action_type": self.CLICK_TYPE_NONE, # Changed key name and value
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
