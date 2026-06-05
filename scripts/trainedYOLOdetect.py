from ultralytics import YOLO
import cv2
import math
import numpy as np
from collections import defaultdict, deque

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

VIOLATION_CLASSES  = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
COMPLIANT_CLASSES  = {"Hardhat", "Safety Vest", "Mask"}

COLOR_VIOLATION = (0,   50,  255)   # red
COLOR_COMPLIANT = (0,  210,   80)   # green
COLOR_OTHER     = (255, 140,   0)   # orange
COLOR_HUD_BG    = (10,   10,  10)

CONF_THRESHOLD  = 0.3
TRAIL_LEN       = 18                # centroid history frames
FLASH_DURATION  = 6                 # frames to flash HUD on new violation

# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def corner_rect(img, x1, y1, x2, y2, color, lw=3, corner_len=22):
    """Corner-tick bounding box with a faint filled overlay."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    l, t = corner_len, lw
    for (px, py, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(img, (px, py), (px + dx * l, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * l), color, t, cv2.LINE_AA)


def draw_label_badge(img, text, x1, y1, color,
                     font=cv2.FONT_HERSHEY_DUPLEX, scale=0.48, thick=1):
    """Pill-shaped label badge above the box."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    px, py = 7, 4
    bx1, by1 = x1, y1 - th - py * 2 - 2
    bx2, by2 = x1 + tw + px * 2, y1 - 2

    # Clamp to frame
    if by1 < 0:
        by1, by2 = y1 + 2, y1 + th + py * 2 + 2

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(img, text, (bx1 + px, by2 - py),
                font, scale, color, thick, cv2.LINE_AA)


def draw_centroid(img, cx, cy, color):
    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
    cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)


def draw_trail(img, pts, color):
    pts = list(pts)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(img, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  HUD overlay
# ─────────────────────────────────────────────

def draw_hud(img, total, violations, compliant, flash):
    """Top-left dark-glass HUD panel."""
    pw, ph = 230, 115
    x0, y0 = 12, 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (60, 60, 60), 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fontM = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(img, "PPE DETECTION", (x0 + 10, y0 + 22),
                font, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(img, (x0 + 10, y0 + 28), (x0 + pw - 10, y0 + 28),
             (60, 60, 60), 1)

    # Detections
    cv2.putText(img, f"Detections", (x0 + 10, y0 + 50),
                font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str(total), (x0 + pw - 40, y0 + 50),
                fontM, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Compliant
    cv2.putText(img, f"Compliant", (x0 + 10, y0 + 72),
                font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str(compliant), (x0 + pw - 40, y0 + 72),
                fontM, 0.75, COLOR_COMPLIANT, 1, cv2.LINE_AA)

    # Violations — flash red when new one appears
    viol_color = (0, 80, 255) if flash else (0, 50, 200)
    cv2.putText(img, f"Violations", (x0 + 10, y0 + 94),
                font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, str(violations), (x0 + pw - 40, y0 + 94),
                fontM, 0.75, viol_color, 1, cv2.LINE_AA)


def draw_status_bar(img, any_violation):
    """Slim status strip at the bottom of the frame."""
    h, w = img.shape[:2]
    bar_h = 32
    color = (0, 30, 180) if any_violation else (0, 130, 40)
    label = "!! VIOLATION DETECTED - ENFORCE PPE COMPLIANCE" if any_violation \
            else ">>  ALL PERSONNEL COMPLIANT"

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.putText(img, label, (14, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

cap   = cv2.VideoCapture('videos/ppe-1-1.mp4')
model = YOLO('Weights/best.pt')
classNames = model.names

# Simple spatial tracker: map bbox-centre hash → trail deque
trails: dict[tuple, deque] = {}

flash_frames    = 0
prev_violations = 0

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    frame_violations = 0
    frame_compliant  = 0
    frame_total      = 0
    any_violation    = False

    # Collect this frame's centroids for trail matching
    current_detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0].item() * 100) / 100
            cls  = int(box.cls[0])
            name = classNames[cls]

            if conf < CONF_THRESHOLD:
                continue

            frame_total += 1
            w, h  = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Classify
            if name in VIOLATION_CLASSES:
                color = COLOR_VIOLATION
                frame_violations += 1
                any_violation = True
            elif name in COMPLIANT_CLASSES:
                color = COLOR_COMPLIANT
                frame_compliant += 1
            else:
                color = COLOR_OTHER

            # Trail key: snap centroid to 30-px grid for stability
            trail_key = (round(cx / 30), round(cy / 30))
            if trail_key not in trails:
                trails[trail_key] = deque(maxlen=TRAIL_LEN)
            trails[trail_key].append((cx, cy))
            current_detections.append(trail_key)

            # Draw
            draw_trail(img, trails[trail_key], color)
            corner_rect(img, x1, y1, x2, y2, color)
            draw_centroid(img, cx, cy, color)

            label = f"{name}  {conf:.2f}"
            draw_label_badge(img, label, x1, y1, color)

    # Prune stale trails
    for key in list(trails):
        if key not in current_detections:
            trails.pop(key, None)

    # Flash HUD when violation count rises
    if frame_violations > prev_violations:
        flash_frames = FLASH_DURATION
    elif flash_frames > 0:
        flash_frames -= 1
    prev_violations = frame_violations

    # Overlays
    draw_hud(img, frame_total, frame_violations, frame_compliant, flash_frames > 0)
    draw_status_bar(img, any_violation)

    cv2.imshow("PPE Detection — Enhanced", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()