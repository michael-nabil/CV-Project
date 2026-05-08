import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(
        self,
        target_size=(64, 64),
        gaussian_kernel=5,
        median_kernel=5,
    ):
        self.target_size = target_size
        self.gaussian_kernel = gaussian_kernel
        self.median_kernel = median_kernel

    # =====================================================
    # MAIN API
    # =====================================================

    def process(
        self,
        image,
        blur="gaussian",
        normalize=True,
    ):

        out = image.copy()

        if blur == "gaussian":
            out = self.gaussian_blur(out)

        elif blur == "median":
            out = self.median_blur(out)

        if normalize:
            out = self.normalize_brightness(out)

        out = self.resize_and_pad(out)

        return out

    # =====================================================
    # OPERATIONS
    # =====================================================

    def gaussian_blur(self, image):

        k = self._ensure_odd(
            self.gaussian_kernel
        )

        return cv2.GaussianBlur(
            image,
            (k, k),
            1.0,
        )

    def median_blur(self, image):

        k = self._ensure_odd(
            self.median_kernel
        )

        return cv2.medianBlur(
            image,
            k,
        )

    def normalize_brightness(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hsv[:, :, 2] = cv2.equalizeHist(
            hsv[:, :, 2]
        )

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

    def resize_and_pad(self, image):

        h, w = image.shape[:2]

        scale = min(
            self.target_size[0] / h,
            self.target_size[1] / w,
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        output = np.zeros(
            (
                self.target_size[0],
                self.target_size[1],
                3,
            ),
            dtype=image.dtype,
        )

        x_off = (
            self.target_size[1] - new_w
        ) // 2

        y_off = (
            self.target_size[0] - new_h
        ) // 2

        output[
            y_off:y_off + new_h,
            x_off:x_off + new_w,
        ] = resized

        return output

    # =====================================================
    # INTERNALS
    # =====================================================

    def _ensure_odd(self, k):

        return k if k % 2 == 1 else k + 1