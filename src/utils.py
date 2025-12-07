import cv2
import numpy as np

def safe_box(box):
    """Normalize and return ints (x,y,w,h) with non-negative coords."""
    x, y, w, h = box
    x, y = int(max(0, x)), int(max(0, y))
    w, h = int(max(0, w)), int(max(0, h))
    return x, y, w, h

def crop_and_resize(img_rgb, box, size=(160,160)):
    x, y, w, h = safe_box(box)
    h_img, w_img = img_rgb.shape[:2]
    # clamp
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)
    crop = img_rgb[y:y2, x:x2]
    if crop.size == 0:
        # return a black image instead of crashing
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size)


