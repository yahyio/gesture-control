# Gesture Control

Hands-free PC control from a webcam using MediaPipe hand tracking. Two modes: volume control and mouse control.

## Features

- **Volume mode**: thumb–index pinch distance maps to system volume with live on-screen meter
- **Mouse mode**: index finger moves cursor, index–middle pinch clicks
- Camera selection at startup
- `m` to switch modes, `Esc` to quit

## Tech Stack

Python · MediaPipe · OpenCV · pycaw · pyautogui

## Run

```bash
pip install -r requirements.txt
python main.py
```

Windows only (system volume API). Requires a webcam or virtual camera (e.g. OBS).
