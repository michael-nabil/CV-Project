import cv2
import numpy as np
from skimage.feature import hog

class FeatureVisualizer:
    @staticmethod
    def get_hog_viz(image):
        """Generates a visual representation of HOG gradients."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # We run HOG again specifically for the image output
        _, hog_image = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True
        )
        return (hog_image * 255).astype(np.uint8)

    @staticmethod
    def get_hsv_channels(image):
        """Splits the image into Hue, Saturation, and Value channels."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return {"h": h, "s": s, "v": v}