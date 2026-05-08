import cv2

from preprocessing.preprocessor import (
    ImagePreprocessor,
)

from features.extractor import (
    FeatureExtractor,
)

from detectors.yolo_detector import (
    YOLODetector,
)

from classifiers.xgb_classifier import (
    XGBClassifier,
)

from pipeline.pipeline import (
    Pipeline,
)

from core.utils import draw_bbox


detector = YOLODetector(
    weights="weights/yolo_best.pt",
)

classifier = XGBClassifier(
    model_path="weights/xgb_model.pkl",
    scaler_path="weights/xgb_scaler.pkl",
    encoder_path="weights/xgb_label_encoder.pkl",
)

preprocessor = ImagePreprocessor(
    target_size=(64, 64),
)

extractor = FeatureExtractor(
    use_hog=True,
    use_hsv=True,
    use_sift=True,
)

pipeline = Pipeline(
    detector=detector,
    classifier=classifier,
    preprocessor=preprocessor,
    extractor=extractor,
)

image = cv2.imread("test.jpg")

results = pipeline.run(image)

for result in results:

    det = result.detection
    pred = result.classification

    label = (
        f"{pred.label} "
        f"{pred.confidence:.2f}"
    )

    image = draw_bbox(
        image,
        det.bbox,
        label=label,
    )

cv2.imshow(
    "Result",
    image,
)

cv2.waitKey(0)
cv2.destroyAllWindows()