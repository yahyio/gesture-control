import math
import os
import subprocess
import time
import urllib.request
from ctypes import POINTER, cast

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

CAM_W, CAM_H = 1280, 720
SMOOTHING = 5
PINCH_CLICK_DIST = 35
FRAME_MARGIN = 120
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()


def pick_camera() -> int:
    print("\nScanning cameras...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()

    if available:
        print("Available camera indices:")
        for i in available:
            print(f"  [{i}]")
    else:
        print("No cameras found.")

    while True:
        try:
            idx = int(input("Select camera index: "))
            if idx >= 0:
                return idx
        except ValueError:
            pass
        print("Please enter a valid number.")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


def get_volume_interface():
    device = AudioUtilities.GetSpeakers()
    raw = device._dev if hasattr(device, '_dev') else device
    interface = raw.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    states = [landmarks[4].x < landmarks[3].x]
    for tip in tips:
        states.append(landmarks[tip].y < landmarks[tip - 2].y)
    return states


def to_pixels(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def draw_hand(frame, landmarks, w, h):
    pts = [to_pixels(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 220, 255), 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def main():
    ensure_model()
    volume = get_volume_interface()
    vol_min, vol_max = volume.GetVolumeRange()[:2]

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cam_index = pick_camera()
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    mode = "volume"
    prev_x, prev_y = 0.0, 0.0
    click_armed = True
    last_frame_time = time.time()
    fps = 0.0

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera not available.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                points = result.hand_landmarks[0]
                draw_hand(frame, points, w, h)
                up = fingers_up(points)

                thumb = to_pixels(points[4], w, h)
                index = to_pixels(points[8], w, h)
                middle = to_pixels(points[12], w, h)

                if mode == "volume":
                    dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                    cv2.line(frame, thumb, index, (80, 220, 255), 3)
                    cv2.circle(frame, thumb, 9, (80, 220, 255), -1)
                    cv2.circle(frame, index, 9, (80, 220, 255), -1)

                    level = np.interp(dist, (30, 260), (0, 100))
                    db = np.interp(dist, (30, 260), (vol_min, vol_max))
                    volume.SetMasterVolumeLevel(float(db), None)

                    bar_y = int(np.interp(level, (0, 100), (440, 160)))
                    cv2.rectangle(frame, (40, 160), (75, 440), (255, 255, 255), 2)
                    cv2.rectangle(frame, (40, bar_y), (75, 440), (80, 220, 255), -1)
                    cv2.putText(frame, f"{int(level)}%", (36, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                else:
                    if up[1] and not up[2]:
                        target_x = np.interp(index[0], (FRAME_MARGIN, w - FRAME_MARGIN), (0, SCREEN_W))
                        target_y = np.interp(index[1], (FRAME_MARGIN, h - FRAME_MARGIN), (0, SCREEN_H))
                        cur_x = prev_x + (target_x - prev_x) / SMOOTHING
                        cur_y = prev_y + (target_y - prev_y) / SMOOTHING
                        pyautogui.moveTo(cur_x, cur_y)
                        prev_x, prev_y = cur_x, cur_y
                        cv2.circle(frame, index, 12, (120, 255, 120), -1)

                    if up[1] and up[2]:
                        dist = math.hypot(index[0] - middle[0], index[1] - middle[1])
                        cv2.line(frame, index, middle, (120, 255, 120), 3)
                        if dist < PINCH_CLICK_DIST and click_armed:
                            pyautogui.click()
                            click_armed = False
                            cv2.circle(frame, middle, 16, (0, 120, 255), -1)
                        elif dist >= PINCH_CLICK_DIST:
                            click_armed = True

                    cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                                  (w - FRAME_MARGIN, h - FRAME_MARGIN), (120, 255, 120), 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last_frame_time, 1e-6))
            last_frame_time = now

            cv2.putText(frame, f"MODE: {mode.upper()}  (m to switch)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"{int(fps)} fps", (w - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

            cv2.imshow("Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("m"):
                mode = "mouse" if mode == "volume" else "volume"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
