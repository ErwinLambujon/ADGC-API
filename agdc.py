from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from functools import lru_cache

import io
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def preload_model():
    load_model()

# ✅ Lazy-load the YOLOv8 model
@lru_cache()
def load_model():
    model = YOLO('best.pt')
    model.eval()
    return model

def render_without_confidence(img, detections, class_names):
    img_with_boxes = img.copy()
    for i in range(len(detections)):
        box = detections.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_with_boxes

def is_valid_ultrasound(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    black_pixel_ratio = np.sum(gray < 30) / gray.size
    if black_pixel_ratio < 0.1:
        return False

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    if np.var(hist_normalized) < 0.0001:
        return False

    return True

@app.post("/detect/")
async def detect_gallstones(file: UploadFile = File(...), return_image: Optional[bool] = True):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image data provided")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    img_array = np.array(image)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if not is_valid_ultrasound(img_array_bgr):
        return JSONResponse(
            content={"error": "The uploaded image does not appear to be an ultrasound"},
            status_code=415
        )

    with torch.no_grad():
        model = load_model()  # ✅ Lazy-load model
        print("🔁 Running model inference...")
        results = model(img_array_bgr, imgsz=640)  # ✅ Force 640x640 input siz
        print("✅ Inference complete")  

    if not results:
        return JSONResponse(
            content={"error": "No results returned from the model"},
            status_code=500
        )

    result = results[0]
    detections = result.boxes
    class_names = result.names

    if len(detections) > 0:
        confidences = detections.conf.cpu().numpy()
        avg_confidence = float(np.mean(confidences) * 100)

        detection_results = {
            "gallstones_detected": True,
            "average_confidence": round(avg_confidence, 2),
            "count": len(detections),
            "detections": []
        }

        for i in range(len(detections)):
            box = detections.xyxy[i].cpu().numpy()
            cls = int(detections.cls[i].item())
            conf = float(detections.conf[i].item())

            detection_results["detections"].append({
                "class": class_names[cls],
                "confidence": round(conf * 100, 2),
                "bbox": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                }
            })

        if return_image:
            detected_img = render_without_confidence(img_array_bgr.copy(), detections, class_names)
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

            detected_pil = Image.fromarray(detected_img)
            img_byte_arr = io.BytesIO()
            detected_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            return Response(
                content=img_byte_arr,
                media_type="image/jpeg", 
                headers={
                    "X-Gallstones-Detected": "true",
                    "X-Average-Confidence": str(round(avg_confidence, 2))
                }
            )
        else:
            return JSONResponse(content=detection_results)
    else:
        return Response(
            status_code=204,
            headers={"X-Message": "No Gallstones Detected"}
        )

@app.post("/detect-json/")
async def detect_gallstones_json(file: UploadFile = File(...)):
    return await detect_gallstones(file, return_image=False)

# ✅ Health check endpoint for testing server status
@app.get("/health")
def health():
    return {"status": "ok"}
