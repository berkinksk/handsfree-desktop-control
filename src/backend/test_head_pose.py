import cv2
import mediapipe as mp
import numpy as np
from detector import HeadEyeDetector
# import csv # For logging data
# import time # For timestamping log entries
# import datetime # For unique log filenames



# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit(1)

# Instantiate detector with updated pose-dependent thresholds (v4)
detector = HeadEyeDetector(
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6,
    norm_pupil_y_diff_threshold_center=0.060,
    norm_pupil_y_diff_threshold_up=0.020,    # Updated to match new default (was 0.030) 
    norm_pupil_y_diff_threshold_down=0.075,   
    norm_pupil_y_diff_threshold_left=0.025,
    norm_pupil_y_diff_threshold_right=0.060,  
    pupil_action_consec_frames=23             
)

# Utilities for drawing
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Use default drawing specifications for face mesh (small thickness for visibility)
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

WINDOW_NAME = "Head Pose & Pupil Action (0.75s Hold)" # Updated window title
print(f"Starting live head pose and pupil action detection (0.75s hold). Press 'q' or close window ({WINDOW_NAME}) to quit.")
# Text overlay will say "Blink!" but is triggered by pupil action
action_text_overlay = "" 

# For debugging pose oscillations
frame_count = 0
prev_pose_label = "center"
pose_oscillations_left_right = 0

# Get frame dimensions for text placement (assuming constant size)
# We get it once before the loop if webcam resolution is fixed,
# or inside if it can change (though less common for this script type)
# For simplicity, let's assume it's fixed after the first frame.
# However, to be robust, it's better to get it each time or ensure it's set.

# Let's get it at the start of the loop to be safe
# img_h, img_w, _ = frame.shape # This was commented out, likely the source of error

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get frame dimensions INSIDE the loop for robustness
    img_h, img_w, _ = frame.shape

    current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Flip the frame for a mirror view (so it feels natural, like looking in a mirror)
    frame = cv2.flip(frame, 1)

    # Get all detection results from the updated method
    detection_results = detector.process_frame_and_detect_features(frame)

    pose_label = detection_results["pose"]
    # Internally this is pupil action, but we call it action_now for display
    action_type = detection_results["action_type"]
    norm_pupil_y_diff = detection_results["norm_pupil_y_diff"]
    landmarks_mp_object = detection_results["landmarks_mp_object"]
    raw_yaw = detection_results["raw_yaw"]
    raw_pitch = detection_results["raw_pitch"]
    raw_roll = detection_results["raw_roll"]
    raw_physical_yaw = detection_results["raw_physical_yaw"]
    raw_physical_pitch = detection_results["raw_physical_pitch"]
    raw_physical_roll = detection_results["raw_physical_roll"]
    # Internally this is total pupil actions, but display as total_actions
    total_actions = detection_results["total_actions"]

    frame_count += 1

    # --- Debugging: Print per-frame info to console ---
    print(f"Frame: {frame_count:04d} | Pose: {pose_label:<6} | NormPupilYDiff: {norm_pupil_y_diff:.3f} | Action: {action_type} | TotalActions: {total_actions} | RawPhysYaw: {raw_physical_yaw:6.1f} | RawPhysPitch: {raw_physical_pitch:6.1f} | RawPhysRoll: {raw_physical_roll:6.1f} | SmoothYaw: {raw_yaw:6.1f} | SmoothPitch: {raw_pitch:6.1f} | SmoothRoll: {raw_roll:6.1f}")

    # --- Debugging: Detect rapid Left/Right oscillations ---
    if (prev_pose_label == "left" and pose_label == "right") or \
       (prev_pose_label == "right" and pose_label == "left"):
        pose_oscillations_left_right += 1
        print(f"RAPID POSE OSCILLATION (L/R): Count {pose_oscillations_left_right}. Prev: {prev_pose_label}, Curr: {pose_label}")
    
    prev_pose_label = pose_label # Update previous pose for next frame

    # Display text says "Blink!" but is triggered by pupil action (action_now)
    if action_type != detector.CLICK_TYPE_NONE:
        action_text_overlay = "Blink!"
        print(f"Action '{action_type}' detected (displayed as Blink!)!") # Updated print
    else:
        if action_text_overlay == "Blink!": 
             action_text_overlay = ""

    # Draw facial landmarks for debugging (using MediaPipe's drawing utils for consistency)
    if landmarks_mp_object:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks_mp_object,
            connections=mp_face_mesh.FACEMESH_TESSELATION, # Using Tesselation for better pupil viz if needed
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )
        # Also draw iris landmarks if refine_landmarks is True (it is)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks_mp_object,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None, # No specific landmark style for irises, just connections
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1) # Blue for irises
        )

        # Draw head pose direction indicator from the nose tip
        # Nose tip landmark is index 1 in Mediapipe Face Mesh
        nose_landmark = landmarks_mp_object.landmark[1]
        nose_x = int(nose_landmark.x * img_w)
        nose_y = int(nose_landmark.y * img_h)

        line_length = 75 # Reduced line length for better visuals when originating from nose
        # Yaw affects horizontal, Pitch affects vertical. Invert pitch for intuitive up/down drawing.
        end_x = int(nose_x + np.sin(raw_yaw * np.pi / 180) * np.cos(raw_pitch * np.pi / 180) * line_length)
        end_y = int(nose_y - np.sin(raw_pitch * np.pi / 180) * line_length)
        cv2.line(frame, (nose_x, nose_y), (end_x, end_y), (255, 0, 0), 3) # Blue line for pose

    # Overlay text: current head pose label
    pose_text_color = (0, 255, 255) # Default yellow
    if pose_label == "right":
        pose_text_color = (0, 0, 255) # Red for Right
    elif pose_label == "left":
        pose_text_color = (0, 0, 255) # Red for Left (can be different if needed)
    elif pose_label == "center":
        pose_text_color = (0, 255, 0) # Green for Center
    elif pose_label == "up":
        pose_text_color = (255, 0, 0) # Blue for Up
    elif pose_label == "down":
        pose_text_color = (255, 0, 0) # Blue for Down (can be different if needed)

    cv2.putText(frame, f"Pose: {pose_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pose_text_color, 2)
    # Display says "Actions" but counts total_actions (pupil based)
    cv2.putText(frame, f"Total Actions: {total_actions}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # PupilYDiff value is NOT displayed on screen as per prior user request
    if action_text_overlay:
        cv2.putText(frame, action_text_overlay, (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Debug info: show raw angles (rounded) for more insight
    cv2.putText(frame, f"Yaw: {raw_yaw:.1f}", (img_w - 170, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {raw_pitch:.1f}", (img_w - 170, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Roll: {raw_roll:.1f}", (img_w - 170, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the result
    cv2.imshow(WINDOW_NAME, frame)
    
    # Exit on 'q' key press or if window is closed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()


print(f"Exiting test_head_pose.py. Total L/R oscillations: {pose_oscillations_left_right}")
