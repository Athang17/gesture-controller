# face_controller_final.py

import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math
import subprocess

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.20  # For double blink
MOUTH_OPEN_THRESHOLD = 0.05
HEAD_TILT_THRESHOLD = 0.03 # How far the nose must move horizontally relative to the chin
ACTION_COOLDOWN = 0.8
DOUBLE_BLINK_WINDOW = 0.5

# --- SETUP ---
keyboard = Controller()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# --- Landmark indices ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LIP_INDICES = [13, 14]
# Points for tilt detection: nose and chin
NOSE_TIP = 1
CHIN_BOTTOM = 152

# --- State Management ---
is_blinking = False
blink_count = 0
last_blink_time = 0
last_action_time = 0

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(eye_indices, landmarks):
    p2, p3, p4, p5, p6 = [landmarks[eye_indices[i]] for i in range(1, 6)]
    p1 = landmarks[eye_indices[0]]
    ver_dist1 = calculate_distance(p2, p6)
    ver_dist2 = calculate_distance(p3, p5)
    hor_dist = calculate_distance(p1, p4)
    return (ver_dist1 + ver_dist2) / (2.0 * hor_dist) if hor_dist != 0 else 0.0

print("Starting Final Face Controller...")
print("- Tilt head to switch apps")
print("- Open mouth for Mission Control")
print("- Double blink for Page Down")
print("- Press 'q' to quit")

# --- MAIN LOOP ---
while True:
    success, img = cap.read()
    if not success: continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(img_rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # --- Calculate all gesture metrics ---
        left_ear = calculate_ear(LEFT_EYE_INDICES, face_landmarks)
        right_ear = calculate_ear(RIGHT_EYE_INDICES, face_landmarks)
        mouth_dist = calculate_distance(face_landmarks[LIP_INDICES[0]], face_landmarks[LIP_INDICES[1]])
        
        # Head tilt detection
        nose_x = face_landmarks[NOSE_TIP].x
        chin_x = face_landmarks[CHIN_BOTTOM].x
        tilt_diff = nose_x - chin_x

        # --- GESTURE MAPPING ---
        if current_time - last_action_time > ACTION_COOLDOWN:
            # 1. Head Tilt Right (Swipe Right)
            if tilt_diff > HEAD_TILT_THRESHOLD:
                print("Head Tilt Right -> Swipe Right")
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 124 using control down'])
                last_action_time = current_time

            # 2. Head Tilt Left (Swipe Left)
            elif tilt_diff < -HEAD_TILT_THRESHOLD:
                print("Head Tilt Left -> Swipe Left")
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 123 using control down'])
                last_action_time = current_time
            
            # 3. Open Mouth for Mission Control
            elif mouth_dist > MOUTH_OPEN_THRESHOLD:
                print("Mouth Open -> Opening Mission Control")
                subprocess.run(["open", "-a", "Mission Control"])
                last_action_time = current_time

        # 4. Double Blink Logic
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            if not is_blinking:
                is_blinking = True
                if current_time - last_blink_time < DOUBLE_BLINK_WINDOW: blink_count += 1
                else: blink_count = 1
                last_blink_time = current_time
        else: is_blinking = False

        if blink_count == 2:
            print("Double Blink -> Scrolling Down")
            keyboard.press(Key.page_down); keyboard.release(Key.page_down)
            blink_count = 0
            last_action_time = current_time

    cv2.imshow("Face Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- CLEANUP ---
print("Stopping controller.")
cap.release()
cv2.destroyAllWindows()
