import cv2
import mediapipe as mp
import numpy as np
from detector import HeadEyeDetector

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit(1)

# Instantiate our head/eye detector with tuned parameters
detector = HeadEyeDetector(min_detection_confidence=0.6,
                           min_tracking_confidence=0.6,
                           blink_threshold=0.32,
                           blink_consec_frames=3)

# Utilities for drawing
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Use default drawing specifications for face mesh (small thickness for visibility)
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

WINDOW_NAME = "Head Pose & Blink Detection"
print(f"Starting live head pose and blink detection. Press 'q' or close window ({WINDOW_NAME}) to quit.")
blink_text_overlay = ""

# For debugging pose oscillations
frame_count = 0
prev_pose_label = "center"
pose_oscillations_left_right = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Flip the frame for a mirror view (so it feels natural, like looking in a mirror)
    frame = cv2.flip(frame, 1)

    # Get all detection results from the new method
    detection_results = detector.process_frame_and_detect_features(frame)

    pose_label = detection_results["pose"]
    blink_now = detection_results["blink_detected"]
    landmarks_mp_object = detection_results["landmarks_mp_object"]
    raw_yaw = detection_results["raw_yaw"]
    raw_pitch = detection_results["raw_pitch"]
    raw_roll = detection_results["raw_roll"]
    total_blinks = detection_results["total_blinks"]
    ear_avg = detection_results["ear_avg"] # Get EAR

    frame_count += 1

    # --- Debugging: Print per-frame info to console ---
    print(f"Frame: {frame_count:04d} | Pose: {pose_label:<6} | EAR: {ear_avg:.2f} | Yaw: {raw_yaw:6.1f} | Pitch: {raw_pitch:6.1f} | Roll: {raw_roll:6.1f}")

    # --- Debugging: Detect rapid Left/Right oscillations ---
    if (prev_pose_label == "left" and pose_label == "right") or \
       (prev_pose_label == "right" and pose_label == "left"):
        pose_oscillations_left_right += 1
        print(f"RAPID POSE OSCILLATION (L/R): Count {pose_oscillations_left_right}. Prev: {prev_pose_label}, Curr: {pose_label}")
    
    prev_pose_label = pose_label # Update previous pose for next frame

    # If a blink was detected on this frame, prepare text to indicate it and print to console
    if blink_now:
        blink_text_overlay = "Blink!"
        print("Blink detected") # Log to console
    else:
        # Clear blink text quickly after showing
        if blink_text_overlay == "Blink!": # Clear only if it was set
             # Keep it on screen for a bit longer using a timer mechanism if desired, 
             # for now, it clears on next frame if no new blink.
             blink_text_overlay = "" 

    # Draw facial landmarks for debugging (using MediaPipe's drawing utils for consistency)
    if landmarks_mp_object:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks_mp_object,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # Draw head pose direction indicator:
    img_h, img_w, _ = frame.shape
    nose_x = int(img_w / 2)
    nose_y = int(img_h / 2)
    line_length = 100
    # Yaw affects horizontal, Pitch affects vertical. Invert pitch for intuitive up/down drawing.
    end_x = int(nose_x + np.sin(raw_yaw * np.pi / 180) * np.cos(raw_pitch * np.pi / 180) * line_length)
    end_y = int(nose_y - np.sin(raw_pitch * np.pi / 180) * line_length)
    cv2.line(frame, (nose_x, nose_y), (end_x, end_y), (255, 0, 0), 3)

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
    # Blink status or count
    cv2.putText(frame, f"Blinks: {total_blinks}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if blink_text_overlay: # Show "Blink!" momentarily
        cv2.putText(frame, blink_text_overlay, (20, 120),
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
