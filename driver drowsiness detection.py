import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import time

# Initialize mixer
mixer.init()
mixer.music.load("music.wav")  # Make sure this file exists in your directory

# EAR function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
EAR_THRESHOLD = 0.25
FRAME_CHECK = 20
COUNTER = 0

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(1)

print("[INFO] Starting webcam. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

    for landmarks in face_landmarks_list:
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        # Convert to numpy array
        left_eye_np = np.array(left_eye)
        right_eye_np = np.array(right_eye)

        # Scale back to original frame size
        left_eye_scaled = left_eye_np * 2
        right_eye_scaled = right_eye_np * 2

        # Calculate EAR
        leftEAR = eye_aspect_ratio(left_eye_scaled)
        rightEAR = eye_aspect_ratio(right_eye_scaled)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.polylines(frame, [left_eye_scaled.astype(np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_scaled.astype(np.int32)], True, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= FRAME_CHECK:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if not mixer.music.get_busy():
                    mixer.music.play()
        else:
            COUNTER = 0
            mixer.music.stop()

        # Show EAR on screen
        cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
mixer.music.stop()
