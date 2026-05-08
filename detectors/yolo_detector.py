# detectors/yolo_detector.py

from ultralytics import YOLO

from core.interfaces import (
    Detector,
    Detection,
)


class YOLODetector(Detector):

    def __init__(
        self,
        weights,
        conf_threshold=0.25,
        iou_threshold=0.45,
    ):
        self.model = YOLO(weights)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.class_names = self.model.names

    def detect(self, image):

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = []

        for box in results.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            conf = float(
                box.conf[0].cpu()
            )

            cls_id = int(
                box.cls[0].cpu()
            )

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    label=self.class_names[
                        cls_id
                    ],
                )
            )

        return detections