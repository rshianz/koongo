import cv2
import mediapipe as mp
import pyautogui
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# configurations 
KEY_LEFT = 'left'
KEY_RIGHT = 'right'
MODIFIER_KEY = 'ctrl'
MODEL_PATH = 'hand_landmarker.task'

# SWIPE SETTING (discrete)
SWIPE_THRESHOLD = 120
SWIPE_COOLDOWN = 1.0  # time to wait after swiping 

# SCROLL SETTINGS (joystick/continues)
# Size of the deadzones in the middle (pixels)
# Hand inside this box = NO SCROLLING
NEUTRAL_ZONE_SIZE = 60

# how fast to scroll? (Lower number = FASTER speed)
# we divide distance by this number to get scroll speed.
SCROLL_DAMPENING = 30

# setup mediapipe 
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)


# helper: finger counter
def get_fingers_status(landmarks):
    fingers = []
    # Index(8) < PIP(6) means UP (Y=0 is top)
    fingers.append(landmarks[8].y < landmarks[6].y)  # Index
    fingers.append(landmarks[12].y < landmarks[10].y)  # Middle
    fingers.append(landmarks[16].y < landmarks[14].y)  # Ring
    fingers.append(landmarks[20].y < landmarks[18].y)  # Pinky
    return fingers


# helper: manual skeleton draw 
def draw_manual_skeleton(img, landmarks):
    h, w, _ = img.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
    ]
    for start_idx, end_idx in connections:
        x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
        x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)


# init camera 
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# state vars
start_x = None
last_swipe_time = 0

print(f"Joystick Scroll Mode Started...")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # flip & convert 
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # detect 
    timestamp_ms = int(time.time() * 1000)
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape
    current_time = time.time()

    screen_cx, screen_cy = w // 2, h // 2

    # draw the natrul zone box for visulity 
    cv2.rectangle(frame,
                  (screen_cx - 100, screen_cy - NEUTRAL_ZONE_SIZE),
                  (screen_cx + 100, screen_cy + NEUTRAL_ZONE_SIZE),
                  (100, 100, 100), 1)

    # reset phase (check for swipe)
    if current_time - last_swipe_time < SWIPE_COOLDOWN:
        remaining = round(SWIPE_COOLDOWN - (current_time - last_swipe_time), 1)
        cv2.putText(frame, f"WAIT... ({remaining}s)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        start_x = None
        cv2.imshow("Koongo : Mac Hand Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        draw_manual_skeleton(frame, hand_landmarks)
        fingers_up = get_fingers_status(hand_landmarks)

        idx_x = int(hand_landmarks[8].x * w)
        idx_y = int(hand_landmarks[8].y * h)
        mid_x = int(hand_landmarks[12].x * w)
        mid_y = int(hand_landmarks[12].y * h)

        #MODE 1: SWIPE (Index finger only) (this emojy = ☝️)
        if fingers_up == [True, False, False, False]:
            cv2.putText(frame, "SWIPE MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(frame, (idx_x, idx_y), 20, (0, 255, 0), cv2.FILLED)

            if start_x is None: start_x = idx_x
            diff_x = idx_x - start_x

            if diff_x > SWIPE_THRESHOLD:
                print("-> RIGHT")
                pyautogui.hotkey(MODIFIER_KEY, KEY_RIGHT)
                last_swipe_time = current_time
            elif diff_x < -SWIPE_THRESHOLD:
                print("<- LEFT")
                pyautogui.hotkey(MODIFIER_KEY, KEY_LEFT)
                last_swipe_time = current_time

            if abs(diff_x) < 20: start_x = idx_x

        #MODE 2: JOYSTICK SCROLL ((Index + Middle) fingers) (this emojy = ✌️) 
        elif fingers_up == [True, True, False, False]:
            cv2.putText(frame, "SCROLL MODE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            start_x = None

            # use center of 2 fingers
            hand_cy = (idx_y + mid_y) // 2
            cv2.circle(frame, (idx_x, hand_cy), 15, (255, 255, 0), cv2.FILLED)

            # calculate distance from center neutral zone
            dist_from_center = hand_cy - screen_cy

            # LOGIC:
                # if dist is positive (Hand Down) -> Scroll Down
                # if dist is negative (Hand Up) -> Scroll Up

            # check if OUTSIDE the neutral zone
            if abs(dist_from_center) > NEUTRAL_ZONE_SIZE:

                # calculate speed base on how far out you are 
                # subtract the neutral zone size so speed starts at 0 at the edge
                effective_dist = abs(dist_from_center) - NEUTRAL_ZONE_SIZE

                # non-linear speed (squaring it makes fast movements much faster)
                scroll_speed = int((effective_dist ** 1.2) / SCROLL_DAMPENING)

                # direction
                if dist_from_center < 0: # hand up 
                    # mac: pos scroll is up
                    pyautogui.scroll(scroll_speed)
                    cv2.line(frame, (screen_cx, screen_cy), (screen_cx, hand_cy), (0, 255, 0), 2)
                else:  # hand down
                    # mac: neg scroll is down
                    pyautogui.scroll(-scroll_speed)
                    cv2.line(frame, (screen_cx, screen_cy), (screen_cx, hand_cy), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "STOP", (idx_x + 20, idx_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            cv2.putText(frame, "IDLE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            start_x = None

    else:
        cv2.putText(frame, "NO HAND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        start_x = None

    cv2.imshow("Koongo : Mac Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
