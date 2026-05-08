import joblib
import numpy as np

from core.interfaces import (
    Classifier,
    ClassificationResult,
)


class XGBClassifier(Classifier):

    def __init__(
        self,
        model_path,
        scaler_path,
        encoder_path,
    ):
        self.model = joblib.load(
            model_path
        )

        self.scaler = joblib.load(
            scaler_path
        )

        self.encoder = joblib.load(
            encoder_path
        )

    def predict(self, features):

        x = self.scaler.transform(
            features.reshape(1, -1)
        )

        pred_id = self.model.predict(x)[0]

        probs = self.model.predict_proba(x)[0]

        label = (
            self.encoder
            .inverse_transform([pred_id])[0]
        )

        confidence = float(
            np.max(probs)
        )

        return ClassificationResult(
            label=label,
            confidence=confidence,
            probabilities=probs,
        )