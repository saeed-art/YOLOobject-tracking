from ultralytics import YOLO
import cv2
import math
import numpy as np
from sort import *
from collections import defaultdict

# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def corner_rect(img, bbox, length=20, thickness=3, rt=1, color_rect=(50, 50, 255), color_corner=(0, 255, 0)):
    """Draw a rectangle with highlighted corner ticks instead of a plain box."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    # Faint filled rectangle for depth
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), color_rect, rt)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Thin border
    cv2.rectangle(img, (x, y), (x2, y2), color_rect, rt)

    # Corner ticks
    l = length
    t = thickness
    # Top-left
    cv2.line(img, (x, y), (x + l, y), color_corner, t)
    cv2.line(img, (x, y), (x, y + l), color_corner, t)
    # Top-right
    cv2.line(img, (x2, y), (x2 - l, y), color_corner, t)
    cv2.line(img, (x2, y), (x2, y + l), color_corner, t)
    # Bottom-left
    cv2.line(img, (x, y2), (x + l, y2), color_corner, t)
    cv2.line(img, (x, y2), (x, y2 - l), color_corner, t)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - l, y2), color_corner, t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color_corner, t)


def draw_id_badge(img, text, pos, font=cv2.FONT_HERSHEY_DUPLEX,
                  font_scale=0.55, thickness=1,
                  bg_color=(30, 30, 30), text_color=(255, 255, 255)):
    """Draw a pill-shaped ID badge."""
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 8, 5
    rx1, ry1 = x, y - th - pad_y * 2
    rx2, ry2 = x + tw + pad_x * 2, y

    # Semi-transparent dark pill
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (100, 100, 100), 1)

    cv2.putText(img, text, (x + pad_x, y - pad_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def conf_color(conf):
    """Map confidence [0,1] → BGR colour: red → yellow → green."""
    if conf >= 0.7:
        return (30, 220, 50)   # green
    elif conf >= 0.5:
        return (30, 200, 220)  # yellow
    else:
        return (30, 80, 220)   # red


def draw_hud_counter(img, count, line_flash):
    """Draw a clean HUD-style vehicle counter in the top-left corner."""
    h, w = img.shape[:2]

    # Background panel
    panel_w, panel_h = 220, 80
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (10, 10), (10 + panel_w, 10 + panel_h), (60, 60, 60), 1)

    # Label
    cv2.putText(img, "VEHICLES COUNTED", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Count value — flash green when a vehicle just crossed
    count_color = (0, 255, 80) if line_flash else (255, 255, 255)
    cv2.putText(img, str(count), (20, 78),
                cv2.FONT_HERSHEY_DUPLEX, 1.7, count_color, 2, cv2.LINE_AA)


def draw_trail(img, trail_points, color=(255, 160, 0)):
    """Draw a fading centroid trail."""
    pts = list(trail_points)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(img, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

cap = cv2.VideoCapture('videos/cars.mp4')

model = YOLO('Weights/yolov8n.pt')
print(f"YOLO is running on: {model.device}") 
classNames = model.names

mask = cv2.imread("images/maskcar.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

# Per-track centroid history for trails  (deque of (cx,cy))
from collections import deque
trail_history = defaultdict(lambda: deque(maxlen=20))

# Confidence per tracked ID (last seen)
id_conf = {}

# Flash timer: frames remaining to show green line
flash_frames = 0
FLASH_DURATION = 8          # frames the line stays green after a crossing

frame_idx = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_idx += 1

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    raw_conf = {}   # bbox-tuple → conf (rough, for later lookup)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf = math.ceil(box.conf[0].item() * 100) / 100
            cls  = int(box.cls[0])
            name = classNames[cls]

            if name in ("car", "truck", "bus", "motorbike") and conf > 0.3:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    resultsTracker = tracker.update(detections)

    # ── Crossing line ──────────────────────────────────────────────
    new_cross = False
    for result in resultsTracker:
        x1, y1, x2, y2, tid = (int(v) for v in result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        trail_history[tid].append((cx, cy))

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if tid not in totalCount:
                totalCount.append(tid)
                new_cross = True

    if new_cross:
        flash_frames = FLASH_DURATION
    elif flash_frames > 0:
        flash_frames -= 1

    # ── Draw line ──────────────────────────────────────────────────
    line_color = (0, 255, 80) if flash_frames > 0 else (0, 0, 255)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), line_color, 4, cv2.LINE_AA)

    # Small label on the line
    cv2.putText(img, "COUNTING LINE", (limits[0], limits[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # ── Draw tracked objects ───────────────────────────────────────
    for result in resultsTracker:
        x1, y1, x2, y2, tid = (int(v) for v in result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Confidence colour (fall back to neutral if not available)
        last_conf = id_conf.get(tid, 0.5)
        box_color = conf_color(last_conf)

        # Trail
        draw_trail(img, trail_history[tid], color=box_color)

        # Corner-tick bounding box
        corner_rect(img, (x1, y1, w, h),
                    length=18, thickness=2,
                    color_rect=box_color, color_corner=box_color)

        # Centroid dot
        cv2.circle(img, (cx, cy), 5, box_color, cv2.FILLED)
        cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)  # white ring

        # ID badge
        badge_text = f"#{tid}"
        draw_id_badge(img, badge_text, (x1, y1), bg_color=(*box_color[:2], 60))

    # ── HUD counter ────────────────────────────────────────────────
    draw_hud_counter(img, len(totalCount), flash_frames > 0)

    cv2.imshow("Vehicle Counter — Enhanced", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()