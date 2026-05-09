import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Standard imports for your modules
from preprocessing.preprocessor import ImagePreprocessor
from features.extractor import FeatureExtractor
from detectors.yolo_detector import YOLODetector
from classifiers.xgb_classifier import XGBClassifier
from pipeline.pipeline import Pipeline
from features.visualizer import FeatureVisualizer
from core.utils import crop_bbox  # Ensure this import exists for the manual loop

app = FastAPI(title="CV Pipeline API", description="YOLO + XGBoost Feature Extraction Pipeline")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialization ---
try:
    detector = YOLODetector(weights="weights/yolo_best.pt")
    classifier = XGBClassifier(
        model_path="weights/xgb_model.pkl",
        scaler_path="weights/xgb_scaler.pkl",
        encoder_path="weights/xgb_label_encoder.pkl"
    )
    pipeline = Pipeline(
        detector=detector,
        classifier=classifier,
        preprocessor=ImagePreprocessor(target_size=(64, 64)),
        extractor=FeatureExtractor(use_hog=True, use_hsv=True)
    )
    print("✅ All models loaded and pipeline is ready.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

def ndarray_to_base64(image):
    """Converts a numpy image array to a base64 string for the frontend."""
    if image is None:
        return ""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # 1. Load the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2. Get Detections from the model
        detections = pipeline.detector.detect(img)
        
        serializable_output = []

        # 3. Manual Pipeline Execution for Visualization
        for det in detections:
            # Step A: Original Crop
            crop = crop_bbox(img, det.bbox)
            if crop is None:
                continue
            
            # Step B: Preprocessing (Filtered)
            processed = pipeline.preprocessor.process(crop)
            
            # Step C: Extraction (For the Model)
            features = pipeline.extractor.extract(processed)
            prediction = pipeline.classifier.predict(features)
            
            # Step D: Visualization (For the Front-end)
            hog_viz = FeatureVisualizer.get_hog_viz(processed)
            hsv_data = FeatureVisualizer.get_hsv_channels(processed)
            
            # Step E: Pack everything for the React Front-end
            item = {
                "detection": {
                    "bbox": [int(x) for x in det.bbox],
                    "confidence": float(det.confidence),
                    "label": str(det.label)
                },
                "classification": {
                    "label": str(prediction.label),
                    "confidence": float(prediction.confidence),
                    "probabilities": prediction.probabilities.tolist() 
                                     if prediction.probabilities is not None else []
                },
                "images": {
                    "origin_crop": ndarray_to_base64(crop),
                    "after_filter": ndarray_to_base64(processed),
                    "hog_viz": ndarray_to_base64(hog_viz),
                    "hsv_h": ndarray_to_base64(hsv_data["h"]),
                    "hsv_s": ndarray_to_base64(hsv_data["s"]),
                    "hsv_v": ndarray_to_base64(hsv_data["v"]),
                },
                "feature_sample": [float(x) for x in features.tolist()[:30]],
                "feature_count": int(len(features))
            }
            serializable_output.append(item)
            
        return {"results": serializable_output}

    except Exception as e:
        # Returns the actual error message to the client for easier debugging
        raise HTTPException(status_code=500, detail=str(e))

# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    # Use reload=True during development so the server restarts when you save code
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)