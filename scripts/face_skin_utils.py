"""Shared OpenCV helpers for face-box and skin-map processing.

Used by the read-only face audit (Step 32) and the separate skin-filter
preprocessing step (Step 33). Geometry helpers stay importable without OpenCV.
"""

from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # Geometry-only imports (tests) still work.
    cv2 = None
    np = None


def expand_box(box, scale, width, height):
    """Expand a face box by scale, clamped to the image."""
    x, y, w, h = box
    extra_x, extra_y = int(round(w * (scale - 1) / 2)), int(round(h * (scale - 1) / 2))
    left, top = max(0, x - extra_x), max(0, y - extra_y)
    right, bottom = min(width, x + w + extra_x), min(height, y + h + extra_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def detect_face(bgr, min_neighbors=5, min_size=30):
    """Detect the largest face using the Haar cascades shipped with OpenCV."""
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    boxes = []
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        for x, y, w, h in cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_size, min_size)):
            boxes.append((int(x), int(y), int(w), int(h)))
        if name == "haarcascade_profileface.xml":
            flipped = cv2.flip(gray, 1)
            for x, y, w, h in cascade.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_size, min_size)):
                boxes.append((int(gray.shape[1] - x - w), int(y), int(w), int(h)))
    return max(boxes, key=lambda box: box[2] * box[3]) if boxes else None


def skin_map(bgr, face_box, cr_range, cb_range, v_min, morph_size):
    """YCrCb skin threshold, restricted to the expanded face when available."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, cr_range[0], cb_range[0]),
                       (255, cr_range[1], cb_range[1]))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_and(mask, cv2.inRange(hsv, (0, 0, v_min), (180, 255, 255)))
    limit = np.zeros(mask.shape, dtype=np.uint8)
    if face_box is not None:
        x, y, w, h = expand_box(face_box, 1.1, bgr.shape[1], bgr.shape[0])
        limit[y:y + h, x:x + w] = 255
    else:
        limit[:] = 255
    mask = cv2.bitwise_and(mask, limit)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def face_skin_region(bgr, args):
    """Return (face_box, face_mask, skin_mask, union_region)."""
    height, width = bgr.shape[:2]
    face_box = detect_face(bgr, args.min_neighbors, args.min_face_size)
    face_mask = np.zeros((height, width), dtype=np.uint8)
    if face_box is not None:
        x, y, w, h = face_box
        face_mask[y:y + h, x:x + w] = 255
    skin = skin_map(bgr, face_box, (args.skin_cr_min, args.skin_cr_max),
                    (args.skin_cb_min, args.skin_cb_max), args.skin_v_min, args.skin_morph)
    return face_box, face_mask, skin, cv2.bitwise_or(face_mask, skin)
