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


def detect_face(bgr, min_neighbors=5, min_size=30, max_size=512):
    """Detect the largest face; detection runs on a downscaled grayscale image."""
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, float(max_size) / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    scaled_min = max(20, int(round(min_size * scale)))
    boxes = []

    def run(name, flip=False):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            return []
        found = []
        targets = [(gray, False)]
        if flip:
            targets.append((cv2.flip(gray, 1), True))
        for image, mirrored in targets:
            for x, y, w, h in cascade.detectMultiScale(
                    image, scaleFactor=1.1, minNeighbors=min_neighbors,
                    minSize=(scaled_min, scaled_min)):
                if mirrored:
                    x = image.shape[1] - x - w
                found.append((int(x), int(y), int(w), int(h)))
        return found

    # Fast path: frontal face usually succeeds, so profile cascades are skipped.
    for name, flip in (("haarcascade_frontalface_default.xml", False),
                       ("haarcascade_frontalface_alt2.xml", False),
                       ("haarcascade_profileface.xml", True)):
        boxes = run(name, flip)
        if boxes:
            break
    if not boxes:
        return None
    box = max(boxes, key=lambda item: item[2] * item[3])
    if scale < 1.0:
        x, y, w, h = box
        box = (int(round(x / scale)), int(round(y / scale)),
               int(round(w / scale)), int(round(h / scale)))
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


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
    face_box = detect_face(bgr, args.min_neighbors, args.min_face_size,
                           getattr(args, "detect_max_size", 512))
    face_mask = np.zeros((height, width), dtype=np.uint8)
    if face_box is not None:
        x, y, w, h = face_box
        face_mask[y:y + h, x:x + w] = 255
    skin = skin_map(bgr, face_box, (args.skin_cr_min, args.skin_cr_max),
                    (args.skin_cb_min, args.skin_cb_max), args.skin_v_min, args.skin_morph)
    return face_box, face_mask, skin, cv2.bitwise_or(face_mask, skin)
