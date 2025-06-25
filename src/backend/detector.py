import cv2
import numpy as np
import mediapipe as mp
import time

class HeadEyeDetector:
    """Detect head pose (orientation) and pupil-based actions (with hold) from video frames using MediaPipe Face Mesh."""
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 norm_pupil_y_diff_threshold_center_left_blink=0.040, # For 'center' pose, left blink
                 norm_pupil_y_diff_threshold_center_right_blink=0.040, # For 'center' pose, right blink
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
        self.norm_pupil_y_diff_thresholds = { # For non-center poses (uses absolute difference)
            # "center" is now handled by specific left/right blink thresholds below
            "up": norm_pupil_y_diff_threshold_up,         # Threshold when head is tilted up
            "down": norm_pupil_y_diff_threshold_down,     # Threshold when head is tilted down
            "left": norm_pupil_y_diff_threshold_left,     # Threshold when head is turned left
            "right": norm_pupil_y_diff_threshold_right    # Threshold when head is turned right
        }
        # Specific thresholds for 'center' pose, distinguishing left/right blinks
        self.norm_pupil_y_diff_threshold_center_left_blink = norm_pupil_y_diff_threshold_center_left_blink
        self.norm_pupil_y_diff_threshold_center_right_blink = norm_pupil_y_diff_threshold_center_right_blink

        # Number of consecutive frames the pupil y-difference must exceed the threshold to trigger a held action.
        # E.g., 23 frames at ~30 FPS is roughly 0.75 seconds.
        # Adjust for desired hold duration before an action is registered.
        self.pupil_action_consec_frames = pupil_action_consec_frames
        self.pupil_action_hold_counter = 0 # Counter for frames norm_pupil_y_diff is above threshold
        self.total_actions = 0     # Total actions detected
        self.action_pending_reset = False # True if action triggered and waiting for condition to cease
        self.current_blink_type_for_hold = None # Stores "LEFT", "RIGHT", or "BOTH" for current held action
        self.last_blink_source_for_double_click = None # Stores "LEFT", "RIGHT", or "BOTH" for the first blink of a potential double click

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
        self.CLICK_TYPE_SINGLE = "SINGLE_CLICK" # Generic single click (e.g., for non-center poses)
        self.CLICK_TYPE_DOUBLE = "DOUBLE_CLICK" # Generic double click (e.g., for non-center poses)
        self.CLICK_TYPE_LEFT_SINGLE = "LEFT_SINGLE_CLICK"
        self.CLICK_TYPE_LEFT_DOUBLE = "LEFT_DOUBLE_CLICK"
        self.CLICK_TYPE_RIGHT_SINGLE = "RIGHT_SINGLE_CLICK"
        
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
        
        left_pupil_lm = landmarks[self.left_pupil_idx]
        right_pupil_lm = landmarks[self.right_pupil_idx]

        left_pupil_y = left_pupil_lm.y * img_h
        right_pupil_y = right_pupil_lm.y * img_h
        
        left_pupil_x = left_pupil_lm.x * img_w
        right_pupil_x = right_pupil_lm.x * img_w

        pupil_x_dist = abs(left_pupil_x - right_pupil_x)
        
        current_norm_pupil_y_diff = 0.0
        if pupil_x_dist > 1e-6: 
            current_norm_pupil_y_diff = (left_pupil_y - right_pupil_y) / pupil_x_dist
        
        action_condition_met = False
        blink_type_this_frame = None # "LEFT" (phys_right), "RIGHT" (phys_left), "BOTH" (non-center)

        # Determine which system-detected blink type occurred based on y-diff and pose
        if pose_label == "center":
            if current_norm_pupil_y_diff > self.norm_pupil_y_diff_threshold_center_left_blink: # System "LEFT" blink
                action_condition_met = True
                blink_type_this_frame = "LEFT" 
            elif current_norm_pupil_y_diff < -self.norm_pupil_y_diff_threshold_center_right_blink: # System "RIGHT" blink
                action_condition_met = True
                blink_type_this_frame = "RIGHT"
        else: # Non-center poses
            current_pose_threshold = self.norm_pupil_y_diff_thresholds.get(pose_label, 0.040) 
            if abs(current_norm_pupil_y_diff) > current_pose_threshold:
                action_condition_met = True
                blink_type_this_frame = "BOTH" 

        action_type_to_return = self.CLICK_TYPE_NONE
        current_time_ns = time.time_ns()

        # --- Timeout logic for a previously started potential double click ---
        # This must be processed BEFORE new blink logic for the current frame,
        # so a timeout can correctly register a single click if the window just expired.
        if self.first_blink_of_potential_double_click_time > 0 and \
           (current_time_ns - self.first_blink_of_potential_double_click_time) > (self.double_click_window_ms * 1_000_000):
            
            timed_out_blink_source = self.last_blink_source_for_double_click # This is "LEFT" for phys. RIGHT, "RIGHT" for phys. LEFT due to mirror
            pending_action_time = self.first_blink_of_potential_double_click_time 

            # Check if enough time has passed since the *start* of the timed-out action to register it
            if (current_time_ns - pending_action_time) > (self.min_time_after_action_ms * 1_000_000): 
                # SWAPPED LOGIC:
                if pose_label == "center" and timed_out_blink_source == "LEFT": # System's "LEFT" blink timed out (Physical RIGHT eye)
                    action_type_to_return = self.CLICK_TYPE_LEFT_SINGLE # Phys. RIGHT eye maps to LEFT_SINGLE_CLICK
                    print(f"Detector: LEFT_SINGLE_CLICK (phys. RIGHT eye, sys LEFT) registered (by timeout).")
                elif pose_label == "center" and timed_out_blink_source == "RIGHT": # System's "RIGHT" blink timed out (Physical LEFT eye)
                    action_type_to_return = self.CLICK_TYPE_RIGHT_SINGLE # Phys. LEFT eye maps to RIGHT_SINGLE_CLICK
                    print(f"Detector: RIGHT_SINGLE_CLICK (phys. LEFT eye, sys RIGHT) registered (by timeout).")
                elif timed_out_blink_source == "BOTH": # Non-center poses, generic single click
                    action_type_to_return = self.CLICK_TYPE_SINGLE
                    print(f"Detector: GENERIC SINGLE_CLICK ({timed_out_blink_source}) registered (by timeout).")
                
                if action_type_to_return != self.CLICK_TYPE_NONE:
                    self.total_actions += 1
                    self.last_blink_time = pending_action_time # Record the time of this confirmed action

            # Reset the double-click tracking variables as the window has expired
            self.first_blink_of_potential_double_click_time = 0 
            self.last_blink_source_for_double_click = None

        # --- Process current frame's physical blink state (hold counter) ---
        if action_condition_met: # A physical blink is currently detected
            if self.action_pending_reset: # An action was just committed, ignore blinks during cooldown
                pass 
            else: # Update hold counter
                if self.pupil_action_hold_counter == 0: # Start of a new hold
                    self.current_blink_type_for_hold = blink_type_this_frame
                    self.pupil_action_hold_counter = 1
                elif self.current_blink_type_for_hold == blink_type_this_frame: # Continuing hold of the same blink type
                    self.pupil_action_hold_counter += 1
                else: # Blink type changed mid-hold, reset
                    self.current_blink_type_for_hold = blink_type_this_frame
                    self.pupil_action_hold_counter = 1
        else: # No physical blink detected in this frame
            if self.action_pending_reset: # Cooldown just finished
                self.action_pending_reset = False
            self.pupil_action_hold_counter = 0 # Reset hold counter
            self.current_blink_type_for_hold = None # Clear current hold type

        # --- Determine click type if a physical blink has been held long enough ---
        if self.pupil_action_hold_counter >= self.pupil_action_consec_frames and not self.action_pending_reset:
            # Check if enough time has passed since the last *committed* action
            if (current_time_ns - self.last_blink_time) > (self.min_time_after_action_ms * 1_000_000):
                confirmed_action_source = self.current_blink_type_for_hold # "LEFT" (phys_right), "RIGHT" (phys_left), or "BOTH"
                
                # Check if this is the second blink of a potential double click
                is_second_blink_of_double = self.first_blink_of_potential_double_click_time > 0 and \
                                           (current_time_ns - self.first_blink_of_potential_double_click_time) <= (self.double_click_window_ms * 1_000_000) and \
                                           self.last_blink_source_for_double_click == confirmed_action_source # Must be the same eye/source

                if is_second_blink_of_double:
                    # This IS the second blink of a potential double click.
                    # SWAPPED LOGIC:
                    if pose_label == "center" and confirmed_action_source == "LEFT": # System's "LEFT" double blink (Physical RIGHT eye)
                        action_type_to_return = self.CLICK_TYPE_LEFT_DOUBLE # Phys. RIGHT eye maps to LEFT_DOUBLE_CLICK
                        print(f"Detector: LEFT_DOUBLE_CLICK (phys. RIGHT eye, sys LEFT) registered.")
                    elif pose_label == "center" and confirmed_action_source == "RIGHT": # System's "RIGHT" double blink (Physical LEFT eye)
                        action_type_to_return = self.CLICK_TYPE_NONE # Phys. LEFT eye double blink is ignored
                        print(f"Detector: Phys. LEFT eye double blink (sys RIGHT) - NO ACTION.")
                    elif confirmed_action_source == "BOTH": # Non-center poses, generic double click
                        action_type_to_return = self.CLICK_TYPE_DOUBLE
                        print(f"Detector: GENERIC DOUBLE_CLICK ({confirmed_action_source}) registered.")
                    
                    if action_type_to_return != self.CLICK_TYPE_NONE:
                        self.total_actions += 1
                    
                    # Reset double-click tracking as it's now resolved (either as double or ignored)
                    self.first_blink_of_potential_double_click_time = 0 
                    self.last_blink_source_for_double_click = None
                else:
                    # This is the FIRST blink of a potential double click (or a single click if no second blink follows)
                    self.first_blink_of_potential_double_click_time = current_time_ns
                    self.last_blink_source_for_double_click = confirmed_action_source
                    # Print statements for first blink:
                    if pose_label == "center":
                        if confirmed_action_source == "LEFT": # Physical RIGHT eye
                            print(f"Detector: Potential first blink (phys. RIGHT eye, sys LEFT) registered. Waiting for double click window or timeout.")
                        elif confirmed_action_source == "RIGHT": # Physical LEFT eye
                            print(f"Detector: Potential first blink (phys. LEFT eye, sys RIGHT) registered. Waiting for double click window or timeout.")
                        else: # Should not happen if logic is correct for center pose
                            print(f"Detector: Potential first blink ({confirmed_action_source}, center) registered. Waiting for double click window or timeout.")
                    else: # Non-center poses
                        print(f"Detector: Potential first blink ({confirmed_action_source}, non-center) registered. Waiting for double click window or timeout.")

                self.last_blink_time = current_time_ns # Record time of this processed blink (start of cooldown for next action)
                self.action_pending_reset = True # Set flag to ignore further blinks until hold counter resets
            else:
                # Not enough time since last action, so reset current hold without action
                self.pupil_action_hold_counter = 0 
                self.current_blink_type_for_hold = None
        
        return action_type_to_return, current_norm_pupil_y_diff

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
