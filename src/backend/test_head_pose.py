import cv2
import numpy as np
import time
import sys
import os

# Adjust the path so we can import the HeadEyeDetector if needed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the project root is in sys.path to allow importing from src package
PROJECT_ROOT = os.path.join(CURRENT_DIR, os.pardir, os.pardir)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the HeadEyeDetector class from the backend module
try:
    from src.backend.detector import HeadEyeDetector
except ImportError:
    # Fallback to relative import if running as a script within the package
    from detector import HeadEyeDetector

def main():
    # Initialize the head-eye detector
    detector = HeadEyeDetector()
    # Open a connection to the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    print("Press 'q' to quit the head pose test.")

    # Main loop to read frames and detect head pose
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame from webcam. Exiting...")
            break

        # Flip the frame horizontally for a mirror view (optional, makes it easier to align movements)
        frame = cv2.flip(frame, 1)

        # Detect head pose
        direction = detector.detect_head_pose(frame)

        # Visual feedback: draw the face box and put text of direction
        face_box = detector.detect_face(frame)
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Pose: {direction.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # (Optional) Draw a line indicating nose direction for debug:
        # We can project a 3D forward vector to see where the nose is pointing.
        # Here we reuse the last rotation from detector (not directly accessible; we could compute again).
        # For simplicity, this step is skipped or could be implemented by recalculating solvePnP here.

        cv2.imshow("Head Pose Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
