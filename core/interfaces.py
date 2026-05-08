from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str
    mask: Optional[np.ndarray] = None


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    probabilities: Optional[np.ndarray] = None


class Detector(ABC):

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        pass


class Classifier(ABC):

    @abstractmethod
    def predict(self, features: np.ndarray) -> ClassificationResult:
        pass