import cv2
import numpy as np
from skimage.feature import hog

class FeatureExtractor:
    def __init__(
        self,
        use_hog=True,
        use_hsv=True,
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        hog_cells_per_block=(2, 2),
        hsv_bins=(8, 8, 8),
    ):
        self.use_hog = use_hog
        self.use_hsv = use_hsv
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hsv_bins = hsv_bins

    def extract(self, image):
        features = []
        if self.use_hog:
            features.append(self.extract_hog(image))
        if self.use_hsv:
            features.append(self.extract_hsv(image))

        return np.concatenate(features).astype(np.float32)

    def extract_hog(self, image):
        gray = self._to_gray(image)
        feat = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return feat.astype(np.float32)

    def extract_hsv(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, self.hsv_bins, [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def _to_gray(self, image):
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image