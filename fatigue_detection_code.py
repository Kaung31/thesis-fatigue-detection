import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) 

# Define facial landmarks
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
MOUTH_LANDMARKS = [61, 291, 78, 308, 14, 18, 81, 311] 

# 3D model points for head pose estimation
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),  # Right eye
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

# Function to estimate head pose (Used for accuracy, not displayed)
def estimate_head_pose(landmarks, frame_width, frame_height):
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),  # Nose tip
        (landmarks[152][0], landmarks[152][1]),  # Chin
        (landmarks[33][0], landmarks[33][1]),  # Left eye corner
        (landmarks[263][0], landmarks[263][1]),  # Right eye corner
        (landmarks[61][0], landmarks[61][1]),  # Left mouth corner
        (landmarks[291][0], landmarks[291][1])  # Right mouth corner
    ], dtype="double")

    focal_length = frame_width
    cam_matrix = np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(FACE_3D_MODEL, image_points, cam_matrix, dist_coeffs)

    if success:
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[1], angles[0], angles[2]  # Return yaw, pitch, roll
    return None, None, None   

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    return (A + B) / (2.0 * C)

# Function to calculate MAR (Mouth Aspect Ratio) for yawning detection
def calculate_mar(mouth_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[mouth_landmarks[1]]) - np.array(landmarks[mouth_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[mouth_landmarks[2]]) - np.array(landmarks[mouth_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[mouth_landmarks[0]]) - np.array(landmarks[mouth_landmarks[3]]))
    return (A + B) / (2.0 * C)

# Function to classify fatigue
def classify_fatigue(slow_blinks, microsleeps, yawns):

    # Priority 1: Microsleeps
    if microsleeps >= 1:
        return "Very Drowsy"

    # Priority 2: Slow Blinks
    if slow_blinks >= 4:
        return "Very Drowsy"
    elif slow_blinks == 3:
        return "Drowsy"
    elif 1 <= slow_blinks <= 2:
        return "Mild Drowsy"

    # Priority 3: Yawns
    if yawns >= 4:
        return "Very Drowsy"
    elif yawns == 3:
        return "Drowsy"
    elif 1 <= yawns <= 2:
        return "Mild Drowsy"

    # Default
    return "Normal"

def is_low_light(frame, threshold=45):  # Lower threshold for extreme low-light conditions

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold

def adjust_brightness(frame, gamma=2.0):  # Stronger gamma correction for near-dark conditions

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # ✅ Apply CLAHE with higher clip limit for better contrast in extreme darkness
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))  
    l = clahe.apply(l)

    # Merge channels and convert back
    lab = cv2.merge((l, a, b))
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ✅ Apply stronger gamma correction for extreme darkness
    gamma_table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    final_frame = cv2.LUT(enhanced_frame, gamma_table)

    # ✅ Apply Gaussian Blur to reduce noise from high enhancement
    final_frame = cv2.GaussianBlur(final_frame, (3, 3), 0)

    return final_frame

def detect_glasses(frame, landmarks):

    eye_indices = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS
    eye_region = [landmarks[i] for i in eye_indices]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for point in eye_region:
        cv2.circle(mask, point, 2, 255, -1)
    mean_brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask)[0]
    
    return mean_brightness > 140  # tweak this threshold as needed

# Blink detection thresholds
EAR_THRESHOLD = 0.2
LEFT_EAR_THRESHOLD = 0.21 # Higher for left eye
RIGHT_EAR_THRESHOLD = 0.21 # Lower for right eye
BLINK_MIN_DURATION = 0.1  # Ignore extremely short blinks
BLINK_NORMAL_MAX = 0.3
BLINK_SLOW_MAX = 0.7
BLINK_MICROSLEEP_MIN = 0.7
MAR_THRESHOLD = 0.6  # Yawning threshold
YAWN_HOLD_TIME = 1.0  # Yawn must last at least 1 second

# Counters
total_blinks = 0
normal_blinks = 0
slow_blinks = 0
microsleeps = 0
yawn_count = 0  

# Tracking Variables
BLINK_START_TIME = None
YAWN_START_TIME = None
blink_type = "No Blink"
blink_active = False
yawn_active = False
yawn_detected = False
yawn_close_time = 0   # ✅ Tracks when mouth closes

# Rolling window setup for fatigue tracking
ROLLING_WINDOW = 60  # Last 60 seconds of data
blink_history = deque(maxlen=ROLLING_WINDOW)
yawn_history = deque(maxlen=ROLLING_WINDOW)

# FPS Calculation Setup
fps_queue = deque(maxlen=30)
fps = 0
start_time = time.time()
last_update_time = int(time.time())  # Track rolling updates

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Only Apply Brightness Adjustment If Lighting is Low
    if is_low_light(frame):
       frame = adjust_brightness(frame)

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    # FPS Calculation (smoother update every 10 frames)
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 0:  
        fps_queue.append(1.0 / elapsed_time)
    if frame_count % 10 == 0:
        fps = sum(fps_queue) / len(fps_queue)
    start_time = current_time

    if results and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
            yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
            if yaw is None or pitch is None:
                yaw, pitch = 0, 0  # Prevent crashes

            # ✅ Compute EAR for Left & Right Eye
            left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
            right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)

            # ✅ Detect glasses
            has_glasses = detect_glasses(frame, landmarks)

            # ✅ Adjust EAR and blink threshold based on yaw and glasses
            if -15 <= yaw <= 15:
                ear = (left_ear + right_ear) / 2
                blink_threshold = EAR_THRESHOLD + (0.1 if has_glasses else 0.0)
            elif yaw > 15:
                ear = left_ear
                blink_threshold = LEFT_EAR_THRESHOLD + (0.1 if has_glasses else 0.0)
            elif yaw < -15:
                ear = right_ear
                blink_threshold = RIGHT_EAR_THRESHOLD + (0.1 if has_glasses else 0.0)

            mar = calculate_mar(MOUTH_LANDMARKS, landmarks)  
            # ✅ Get MAR threshold dynamically based on yaw angle         
            if yaw < -30:
                mar_threshold = MAR_THRESHOLD + 0.04
            else:
                mar_threshold = MAR_THRESHOLD

            
            # ✅ Blink Detection (Independent)
            if ear < blink_threshold and not blink_active:
                BLINK_START_TIME = time.perf_counter()
                blink_active = True
            elif ear >= blink_threshold and blink_active:
                blink_duration = time.perf_counter() - BLINK_START_TIME
                blink_active = False  

                # ✅ Classify Blink Type (Prevents Double Counting)
                if blink_duration >= BLINK_MICROSLEEP_MIN:
                    blink_type = "Microsleep (Very Drowsy!)"
                    microsleeps += 1
                elif BLINK_SLOW_MAX > blink_duration >= BLINK_NORMAL_MAX:
                    blink_type = "Slow Blink (Drowsy)"
                    slow_blinks += 1
                elif BLINK_MIN_DURATION < blink_duration < BLINK_NORMAL_MAX:
                    blink_type = "Normal Blink"
                    normal_blinks += 1
            
            # ✅ Compute Total Blinks Dynamically
            total_blinks = normal_blinks + slow_blinks + microsleeps

            # ✅ Yawn Detection (Independent from Blink)
            if mar > mar_threshold and not yawn_active:
                YAWN_START_TIME = time.perf_counter()
                yawn_active = True
                yawn_detected = False  # Reset flag when mouth starts opening
            elif mar <= mar_threshold and yawn_active:
                yawn_close_time = time.perf_counter()
                yawn_active = False

                # ✅ Ensure yawn lasts long enough before counting
                if (yawn_close_time - YAWN_START_TIME) >= YAWN_HOLD_TIME and not yawn_detected:
                    yawn_count += 1  # ✅ Count yawn only once per full cycle
                    yawn_detected = True  # Prevent multiple counts

            # ✅ Reset yawn detection after full cycle
            if yawn_detected and (time.perf_counter() - yawn_close_time) >= 1.0:
                yawn_detected = False  # Allow a new yawn to be detected

            # Update rolling window every second
            current_time_sec = int(time.time())
            if current_time_sec != last_update_time:
                blink_history.append(total_blinks)
                yawn_history.append(yawn_count)
                last_update_time = current_time_sec

            fatigue_level = classify_fatigue(sum(blink_history), slow_blinks, microsleeps, sum(yawn_history), yaw, pitch)

        cv2.putText(frame, f"Fatigue Level: {fatigue_level}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Blink: {blink_type}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Blinks: {total_blinks}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Normal: {normal_blinks}, Slow: {slow_blinks}, Micro: {microsleeps}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawn Status: {yawn_active}", (30, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink Status: {blink_active}", (30, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
