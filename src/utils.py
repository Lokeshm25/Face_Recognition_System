import cv2
import numpy as np

def safe_box(box):
    """
    Normalize and clamp bounding box:
    input: [x, y, w, h] (may contain negatives)
    output: (x, y, w, h) all >= 0 as ints
    """
    x, y, w, h = box
    x, y = int(max(0, x)), int(max(0, y))
    w, h = int(max(0, w)), int(max(0, h))
    return x, y, w, h

def crop_and_resize(img_rgb, box, size=(160, 160)):
    """
    Crops image using bounding box and resizes to given size.
    If crop is invalid / empty, returns a black image instead of crashing.
    """
    x, y, w, h = safe_box(box)
    h_img, w_img = img_rgb.shape[:2]

    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    crop = img_rgb[y:y2, x:x2]
    if crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    return cv2.resize(crop, size)
