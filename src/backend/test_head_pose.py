import cv2
import sys
import os

# Ensure the project src path is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.detector import HeadEyeDetector

def main():
    det = HeadEyeDetector()
    # Confirm that the face mesh model loaded
    print(f"Face Mesh loaded: {det.mp_face_mesh is not None}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    cv2.namedWindow("Head Pose Test", cv2.WINDOW_NORMAL)
    frame_count = 0
    last_pose = None
    last_face_detected = None
    last_blink = False  # track blink state

    while True:
        # Break loop if window is closed
        if cv2.getWindowProperty("Head Pose Test", cv2.WND_PROP_VISIBLE) < 1:
            break
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        frame_count += 1

        # Run full head-pose and blink detection pipeline
        result = det.process_frame(frame)
        pose_label = result['pose']
        blink_detected = result['blink']
        face_box = det.detect_face(frame)
        face_detected = face_box is not None

        # Log any changes in face presence or pose
        if face_detected != last_face_detected or pose_label != last_pose:
            print(f"[Frame {frame_count}] Face detected: {face_detected}, Pose: {pose_label}")
            last_face_detected = face_detected
            last_pose = pose_label

        # Log blink events when they occur
        if blink_detected and not last_blink:
            print(f"[Frame {frame_count}] Blink detected")
        last_blink = blink_detected

        # Draw a rectangle around the face if detected
        if face_detected:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Annotate the pose label on the frame
        cv2.putText(frame, f"Pose: {pose_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Annotate blink status on the frame
        cv2.putText(frame, f"Blink: {blink_detected}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Also display the raw pitch and yaw angles for reference
        if hasattr(det, 'last_raw_pitch') and hasattr(det, 'last_raw_yaw'):
            angles_text = f"Pitch: {det.last_raw_pitch:.1f}°  Yaw: {det.last_raw_yaw:.1f}°"
            cv2.putText(frame, angles_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Head Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
