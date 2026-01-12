from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import tempfile
import logging
from inference import LungCancerSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Cancer Segmentation API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize segmentation model
segmentation_model = None

@app.on_event("startup")
async def startup_event():
    """Load the nnU-Net model on startup"""
    global segmentation_model
    try:
        segmentation_model = LungCancerSegmentation()
        logger.info("nnU-Net model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue without model for development

@app.get("/")
async def root():
    return {"message": "Lung Cancer Segmentation API", "status": "running"}

@app.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    """
    Predict lung cancer segmentation from uploaded CT image
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if segmentation_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_input:
            content = await file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # Run segmentation
        output_path = segmentation_model.predict(temp_input_path)
        
        # Clean up input file
        os.unlink(temp_input_path)
        
        # Return segmented image
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="segmentation_result.png"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)