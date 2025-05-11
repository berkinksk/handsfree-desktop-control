import cv2
import sys
import os
# ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.detector import HeadEyeDetector

def main():
    det = HeadEyeDetector()
    # debug: report model load status
    print(f"Face cascade loaded: {not det.face_cascade.empty()}")
    print(f"Landmark model loaded: {det.landmark_model is not None}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # create window once for testing (allows visibility check)
    cv2.namedWindow("Head Pose Test", cv2.WINDOW_NORMAL)

    # speed optimization and informative logging
    resize_width = 320
    last_pose = None
    last_face_detected = None
    frame_count = 0

    # loop for real-time testing
    while True:
        # exit if window was closed by user
        if cv2.getWindowProperty("Head Pose Test", cv2.WND_PROP_VISIBLE) < 1:
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # increment frame counter
        frame_count += 1

        # resize for faster detection
        h_o, w_o = frame.shape[:2]
        scale = resize_width / float(w_o)
        h_s = int(h_o * scale)
        frame_small = cv2.resize(frame, (resize_width, h_s))

        # run detection on resized frame
        face_small = det.detect_face(frame_small)
        # detect landmarks on resized frame
        landmarks_small = det.detect_landmarks(frame_small, face_small)
        # estimate pose on full-resolution frame for more accurate angles
        pose = det.detect_head_pose(frame)
        face_detected = face_small is not None

        # log landmarks if found, flatten nested lists robustly
        if landmarks_small:
            # flatten landmarks list
            flat_landmarks = []
            for pt in landmarks_small:
                if isinstance(pt, (list, tuple)) and pt and isinstance(pt[0], (list, tuple)):
                    flat_landmarks.extend(pt)
                else:
                    flat_landmarks.append(pt)
            print(f"[Frame {frame_count}] Landmarks detected: {len(flat_landmarks)} points")
            # draw up to first 50 landmarks on original frame
            for lx, ly in flat_landmarks[:50]:
                x_l = int(lx / scale)
                y_l = int(ly / scale)
                cv2.circle(frame, (x_l, y_l), 2, (255, 0, 0), -1)

        # log when detection or pose changes
        if face_detected != last_face_detected or pose != last_pose:
            print(f"[Frame {frame_count}] Face detected: {face_detected}, Pose: {pose}")
            last_face_detected, last_pose = face_detected, pose

        # map face coords back to original frame and annotate
        if face_detected:
            x_s, y_s, w_s, h_s = face_small
            x = int(x_s / scale)
            y = int(y_s / scale)
            w = int(w_s / scale)
            h = int(h_s / scale)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # annotate pose on original frame
        cv2.putText(
            frame,
            f"Pose: {pose}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # show results on original frame
        cv2.imshow("Head Pose Test", frame)
        # exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 