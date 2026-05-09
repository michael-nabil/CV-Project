from dataclasses import dataclass
from typing import List, Any
import numpy as np
from core.utils import crop_bbox

@dataclass
class PipelineResult:
    detection: Any
    classification: Any
    features: np.ndarray      
    preprocessed_image: np.ndarray  

class Pipeline:
    def __init__(self, detector, classifier, preprocessor, extractor):
        self.detector = detector
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.extractor = extractor

    def run(self, image: np.ndarray) -> List[PipelineResult]:
        detections = self.detector.detect(image)
        results = []

        for det in detections:
            crop = crop_bbox(image, det.bbox)
            if crop is None:
                continue

            processed = self.preprocessor.process(crop)

            features = self.extractor.extract(processed)

            prediction = self.classifier.predict(features)

            results.append(
                PipelineResult(
                    detection=det,
                    classification=prediction,
                    features=features,
                    preprocessed_image=processed
                )
            )
        return results