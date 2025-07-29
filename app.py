import io
import json
import logging
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """Defines the structure for the prediction response."""
    prediction: str = Field(..., description="The predicted class label (e.g., 'Cat' or 'Dog').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The model's confidence score (0.0 to 1.0).")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Dog",
                "confidence": 0.987
            }
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cat vs. Dog Classifier API",
    description="An API to predict whether an image contains a cat or a dog using a trained Keras model.",
    version="1.0.0"
)

# --- Globals for Model and Labels ---
model = None
labels = None

# --- Startup Event to Load Model and Labels ---
@app.on_event("startup")
async def startup_event():
    """
    Load the Keras model and class labels on application startup.
    """
    global model, labels
    try:
        # Load the trained Keras model
        model = load_model('models/best_model.h5')
        logger.info("✅ Keras model loaded successfully.")

        # Load the class labels
        with open('models/labels.json', 'r') as f:
            # Convert string keys from JSON back to integers
            labels_str_keys = json.load(f)
            labels = {int(k): v for k, v in labels_str_keys.items()}
        logger.info("✅ Class labels loaded successfully.")
        
    except Exception as e:
        logger.error(f"❌ Error loading model or labels: {e}")
        # In a real application, you might want the app to fail to start
        # if the model can't be loaded.
        model = None
        labels = None

# --- Helper Function for Image Preprocessing ---
def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Preprocesses the image for model prediction.
    - Converts to RGB if needed.
    - Resizes the image.
    - Normalizes pixel values.
    - Adds a batch dimension.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- API Endpoints ---
@app.get("/")
async def root():
    """
    Root endpoint with a welcome message.
    """
    return {"message": "Welcome to the Cat vs. Dog Classifier API! Visit /docs for more info."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, preprocesses it, and returns the model's prediction.
    """
    if not model or not labels:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")
    
    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(128, 128))
        
        # Make prediction
        prediction_prob = model.predict(processed_image)[0][0]
        
        # Interpret the prediction from the sigmoid output
        if prediction_prob > 0.5:
            predicted_index = 1  # Corresponds to 'Dog'
            confidence = prediction_prob
        else:
            predicted_index = 0  # Corresponds to 'Cat'
            confidence = 1 - prediction_prob
            
        predicted_label = labels.get(predicted_index, "Unknown")
        
        return {
            "prediction": predicted_label,
            "confidence": float(confidence)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)