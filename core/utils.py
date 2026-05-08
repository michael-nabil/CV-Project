
import cv2
import numpy as np


def crop_bbox(
    image,
    bbox,
    min_size=5,
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if (x2 - x1) < min_size:
        return None

    if (y2 - y1) < min_size:
        return None

    return image[y1:y2, x1:x2]


def draw_bbox(
    image,
    bbox,
    label=None,
    color=(0, 255, 0),
    thickness=2,
):

    out = image.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        out,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    if label:

        cv2.putText(
            out,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return out