"""
FastAPI backend for AI vs Real Image Detector
Provides prediction endpoint with Grad-CAM visualization
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from utils.preprocess import preprocess_image
from utils.gradcam import generate_gradcam

# Initialize FastAPI app
app = FastAPI(title="AI vs Real Image Detector", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
autoencoder_model = None
threshold = None

def load_model():
    """Load the trained autoencoder model and threshold"""
    global autoencoder_model, threshold
    
    model_path = "model/ai_detector_model.h5"
    threshold_path = "model/threshold.npy"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    autoencoder_model = tf.keras.models.load_model(model_path)
    
    if os.path.exists(threshold_path):
        threshold = float(np.load(threshold_path))
    else:
        # Default threshold if not found
        threshold = 0.1
        print(f"Warning: Threshold file not found. Using default threshold: {threshold}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        print("Please run 'python train_model.py' first to train the model.")

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI vs Real Image Detector API is running"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is AI-generated or real
    Returns prediction, confidence, and Grad-CAM heatmap
    """
    if autoencoder_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image for model
        processed_image = preprocess_image(image)
        
        # Get reconstruction error
        reconstruction = autoencoder_model.predict(processed_image, verbose=0)
        reconstruction_error = np.mean(np.square(processed_image - reconstruction))
        
        # Determine prediction based on threshold
        # Higher reconstruction error = more likely AI-generated
        is_ai_generated = reconstruction_error > threshold
        prediction = "AI-generated" if is_ai_generated else "Real"
        
        # Calculate confidence (distance from threshold)
        confidence = min(abs(reconstruction_error - threshold) / threshold, 1.0)
        
        # Generate Grad-CAM heatmap
        heatmap_b64 = generate_gradcam(autoencoder_model, processed_image)
        
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": float(confidence),
            "reconstruction_error": float(reconstruction_error),
            "threshold": float(threshold),
            "heatmap": heatmap_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
