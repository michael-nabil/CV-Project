from dataclasses import dataclass
from pyexpat import features

from core.utils import crop_bbox


@dataclass
class PipelineResult:
    detection: object
    classification: object
    # features: object 
    # preprocessed: object


class Pipeline:

    def __init__(
        self,
        detector,
        classifier,
        preprocessor,
        extractor,
    ):
        self.detector = detector
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.extractor = extractor

    def run(self, image):

        detections = (
            self.detector.detect(image)
        )

        results = []

        for det in detections:

            crop = crop_bbox(
                image,
                det.bbox,
            )

            if crop is None:
                continue

            processed = (
                self.preprocessor.process(
                    crop
                )
            )

            features = (
                self.extractor.extract(
                    processed
                )
            )

            prediction = (
                self.classifier.predict(
                    features
                )
            )

            results.append(
                PipelineResult(
                    detection=det,
                    classification=prediction,
                )
            )

        return results