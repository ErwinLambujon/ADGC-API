from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from typing import Optional
import io
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from ultralytics import YOLO

app = FastAPI()

# Load your YOLOv8 model (assuming this is already done in your application)
model = YOLO('best.pt')
model.eval()

def render_without_confidence(img, detections, class_names):
    """Draw bounding boxes on image without confidence scores or labels"""
    # Clone the image to avoid modifying the original
    img_with_boxes = img.copy()
    
    # Process each detection
    for i in range(len(detections)):
        box = detections.xyxy[i].cpu().numpy()  # Get bounding box coordinates
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Draw bounding box only - no label
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_with_boxes

def is_valid_ultrasound(img_array):
    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Check 1: Ultrasound images typically have black borders/regions
    black_pixel_ratio = np.sum(gray < 30) / gray.size
    if black_pixel_ratio < 0.1:  # Ultrasounds typically have some black regions
        return False
    
    # Check 2: Check for brightness distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    
    # Ultrasounds typically have specific brightness distribution
    if np.var(hist_normalized) < 0.0001:  # Too uniform distribution
        return False
    
    return True

@app.post("/detect/")
async def detect_gallstones(file: UploadFile = File(...), return_image: Optional[bool] = True):
    # Validate and read uploaded image
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Convert to numpy array and change color format
    img_array = np.array(image)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Check if image appears to be an ultrasound before running the model
    if not is_valid_ultrasound(img_array_bgr):
        # Not a valid ultrasound image - just return error without image
        return JSONResponse(
            content={"error": "The uploaded image does not appear to be an ultrasound"},
            status_code=415
        )
    
    # Only run the model if we believe it's an ultrasound
    with torch.no_grad():
        results = model(img_array_bgr)
    
    # Process valid ultrasound image
    if results:
        result = results[0]
        detections = result.boxes
        class_names = result.names
        
        if len(detections) > 0:
            # Calculate average confidence
            confidences = detections.conf.cpu().numpy()
            avg_confidence = float(np.mean(confidences) * 100)
            
            # Process detection results
            detection_results = {
                "gallstones_detected": True,
                "average_confidence": round(avg_confidence, 2),
                "count": len(detections),
                "detections": []
            }
            
            # Add details for each detection
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
                # Draw bounding boxes on image (without text labels)
                detected_img = render_without_confidence(img_array_bgr.copy(), detections, class_names)
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to bytes
                detected_pil = Image.fromarray(detected_img)
                img_byte_arr = io.BytesIO()
                detected_pil.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Return image with detections
                return Response(
                    content=img_byte_arr, 
                    media_type="image/jpeg", 
                    headers={
                        "X-Gallstones-Detected": "true",
                        "X-Average-Confidence": str(avg_confidence)
                    }
                )
            else:
                return JSONResponse(content=detection_results)
        else:
            detection_results = {
                "gallstones_detected": False,
                "message": "No Gallstones Detected"
            }
            
            if return_image:
                # Convert original image to bytes
                detected_img = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
                detected_pil = Image.fromarray(detected_img)
                img_byte_arr = io.BytesIO()
                detected_pil.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                return Response(
                    content=img_byte_arr, 
                    media_type="image/jpeg",
                    headers={"X-Gallstones-Detected": "false", "X-Message": "No Gallstones Detected"}
                )
            else:
                return JSONResponse(content=detection_results)
    else:
        detection_results = {
            "error": "No results returned from the model"
        }
        
        if return_image:
            detected_img = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
            detected_pil = Image.fromarray(detected_img)
            img_byte_arr = io.BytesIO()
            detected_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(
                content=img_byte_arr, 
                media_type="image/jpeg",
                headers={"X-Error": "No results returned from the model"}
            )
        else:
            return JSONResponse(content=detection_results)
    # Validate and read uploaded image
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Convert to numpy array and change color format
    img_array = np.array(image)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Check if image appears to be an ultrasound before running the model
    if not is_valid_ultrasound(img_array_bgr):
        # Not a valid ultrasound image
        if return_image:
            # Return the original image with error headers
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(
                content=img_byte_arr,
                media_type="image/jpeg",
                status_code=415,  # Unsupported Media Type
                headers={"X-Error": "Not an ultrasound image"}
            )
        else:
            return JSONResponse(
                content={"error": "The uploaded image does not appear to be an ultrasound"},
                status_code=415
            )
    
    # Only run the model if we believe it's an ultrasound
    with torch.no_grad():
        results = model(img_array_bgr)
    
    # Continue with the rest of your code...
    # Validate file is not empty
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    # Read uploaded image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Convert to numpy array and change color format
    img_array = np.array(image)
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Check if image is valid for our model
    is_valid, results = is_valid_ultrasound(img_array_bgr)
    
    if not is_valid:
        # Not a valid ultrasound image
        if return_image:
            # Return the original image with error headers
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(
                content=img_byte_arr,
                media_type="image/jpeg",
                status_code=415,  # Unsupported Media Type
                headers={"X-Error": "Not an ultrasound image"}
            )
        else:
            return JSONResponse(
                content={"error": "The uploaded image does not appear to be an ultrasound"},
                status_code=415
            )
    
    # Process valid ultrasound image
    if results:
        result = results[0]
        detections = result.boxes
        class_names = result.names
        
        if len(detections) > 0:
            # Calculate average confidence
            confidences = detections.conf.cpu().numpy()
            avg_confidence = float(np.mean(confidences) * 100)
            
            # Process detection results
            detection_results = {
                "gallstones_detected": True,
                "average_confidence": round(avg_confidence, 2),
                "count": len(detections),
                "detections": []
            }
            
            # Add details for each detection
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
                # Draw bounding boxes on image (without text labels)
                detected_img = render_without_confidence(img_array_bgr.copy(), detections, class_names)
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to bytes
                detected_pil = Image.fromarray(detected_img)
                img_byte_arr = io.BytesIO()
                detected_pil.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Return image with detections
                return Response(
                    content=img_byte_arr, 
                    media_type="image/jpeg", 
                    headers={
                        "X-Gallstones-Detected": "true",
                        "X-Average-Confidence": str(avg_confidence)
                    }
                )
        else:
            detection_results = {
                "gallstones_detected": False,
                "message": "No Gallstones Detected"
            }
            
            if return_image:
                # Convert original image to bytes
                detected_img = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
                detected_pil = Image.fromarray(detected_img)
                img_byte_arr = io.BytesIO()
                detected_pil.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                return Response(
                    content=img_byte_arr, 
                    media_type="image/jpeg",
                    headers={"X-Gallstones-Detected": "false"}
                )
    else:
        detection_results = {
            "error": "No results returned from the model"
        }
        
        if return_image:
            detected_img = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
            detected_pil = Image.fromarray(detected_img)
            img_byte_arr = io.BytesIO()
            detected_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return Response(
                content=img_byte_arr, 
                media_type="image/jpeg",
                headers={"X-Error": "No results returned from the model"}
            )
    
    # Return JSON if not returning image
    if not return_image:
        return JSONResponse(content=detection_results)

# Add endpoint to get just the JSON results without the image
@app.post("/detect-json/")
async def detect_gallstones_json(file: UploadFile = File(...)):
    return await detect_gallstones(file, return_image=False)