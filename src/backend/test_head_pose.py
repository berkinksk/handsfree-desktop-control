import cv2
import sys
import os
# ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.detector import HeadEyeDetector

def main():
    det = HeadEyeDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read frame")
        return

    # run detection
    face = det.detect_face(frame)
    pose = det.detect_head_pose(frame)

    # annotate frame
    if face is not None:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"Pose: {pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show results
    cv2.imshow("Head Pose Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print to console as well
    print(f"Estimated head pose: {pose}")

if __name__ == '__main__':
    main() 