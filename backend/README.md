# Lung Cancer Segmentation Backend

FastAPI backend for multi-class lung cancer segmentation using nnU-Net 2D.

## Setup

1. Create virtual environment:
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
cd src
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /predict
Upload a lung CT image (JPG) and get segmentation result.

**Request:**
- File: JPG image of lung CT scan

**Response:**
- PNG image with color-coded segmentation overlay

**Classes:**
- 0: Background (Black)
- 1: Adenocarcinoma (Red)
- 2: Large Cell Carcinoma (Green)  
- 3: Squamous Cell Carcinoma (Blue)

### GET /
Health check endpoint.

## Model Training

To train your own nnU-Net model:

1. Prepare training data in nnU-Net format
2. Run preprocessing and training:
```bash
python train_nnunet.py
```

## Development Mode

The system includes fallback functionality when nnU-Net models are not available, generating synthetic segmentation masks for development and testing.

## File Structure

- `main.py` - FastAPI application
- `inference.py` - Segmentation pipeline
- `model_loader.py` - nnU-Net model loading
- `utils.py` - Image processing utilities
- `convert_jpg_to_nifti.py` - Format conversion
- `validate_masks.py` - Mask validation
- `train_nnunet.py` - Training setup



set nnUNet_raw=%cd%\data\nnunet\nnUNet_raw
set nnUNet_preprocessed=%cd%\data\nnunet\nnUNet_preprocessed
set nnUNet_results=%cd%\data\nnunet\nnUNet_results



Why nnU‑Net CLI is the correct choice
nnU‑Net’s training command:

nnUNetv2_train 1 2d nnUNetTrainer 0
already includes:

✅ Best‑known U‑Net architecture

✅ Dice + CE loss (correct for segmentation)

✅ Automatic batch size

✅ Automatic patch size

✅ Multi‑class handling

✅ 5‑fold cross‑validation

✅ Ensemble support

✅ Proven results (top in medical challenges)