# Lung Cancer Segmentation using nnU-Net 2D

A complete full-stack application for multi-class lung cancer segmentation using deep learning.

## Overview

This project provides an end-to-end solution for segmenting lung cancer types from CT images using nnU-Net 2D architecture. Users can upload JPG images and receive color-coded segmentation results identifying different cancer types.

### Cancer Classes
- **Background** (Class 0): Black
- **Adenocarcinoma (ADC)** (Class 1): Red  
- **Large Cell Carcinoma (LCC)** (Class 2): Green
- **Squamous Cell Carcinoma (SCC)** (Class 3): Blue

## Tech Stack

**Frontend:**
- React 18 with Vite
- JavaScript
- Axios for API communication
- Plain CSS styling

**Backend:**
- Python 3.9+
- FastAPI
- nnU-Net v2 (2D)
- PyTorch
- OpenCV, NumPy, Pillow

## Project Structure

```
lung-cancer-segmentation/
├── backend/
│   ├── src/                    # All Python files (single venv)
│   │   ├── main.py            # FastAPI application
│   │   ├── inference.py       # Segmentation pipeline
│   │   ├── model_loader.py    # nnU-Net model loading
│   │   ├── utils.py           # Image processing utilities
│   │   ├── convert_jpg_to_nifti.py
│   │   ├── validate_masks.py
│   │   └── train_nnunet.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── api.js             # Backend API calls
│   │   ├── index.css          # Styling
│   │   └── main.jsx           # React entry point
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── nnunet/
│       └── Dataset001_LungCancer/
└── README.md
```

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cd src
python main.py
```

Backend runs on: `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

### 3. Usage

1. Open `http://localhost:5173` in your browser
2. Upload a lung CT image (JPG format)
3. Click "Segment Tumor" 
4. View the color-coded segmentation result

## API Endpoints

### POST /predict
- **Input**: JPG image file
- **Output**: PNG image with segmentation overlay
- **Timeout**: 60 seconds

### GET /
- Health check endpoint

## Development Features

- **Fallback Mode**: Works without trained models (generates synthetic results)
- **Error Handling**: Comprehensive error messages and validation
- **CORS Enabled**: Frontend-backend integration
- **Responsive UI**: Works on desktop and mobile
- **Loading States**: Visual feedback during processing

## Model Training

To train your own nnU-Net model:

1. Prepare training data in nnU-Net format
2. Place images in `data/nnunet/Dataset001_LungCancer/imagesTr/`
3. Place labels in `data/nnunet/Dataset001_LungCancer/labelsTr/`
4. Run training setup:

```bash
cd backend/src
python train_nnunet.py
```

5. Execute nnU-Net commands:
```bash
nnUNetv2_plan_and_preprocess -d 1
nnUNetv2_train 1 2d 0
```

## File Naming Convention

**Training Images:** `LungCancer_001_0000.nii.gz`
**Training Labels:** `LungCancer_001.nii.gz`

## Requirements

- Python 3.9+
- Node.js 16+
- 8GB+ RAM recommended
- GPU optional (CUDA support)

## Troubleshooting

### Backend Issues
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify CORS settings for frontend communication

### Frontend Issues  
- Clear browser cache
- Check console for JavaScript errors
- Ensure backend is running on port 8000

### Model Issues
- System works in fallback mode without trained models
- For production, train nnU-Net model with your dataset
- Check nnU-Net installation and environment variables

## License

Research Use Only

## Contributing

1. Follow the existing code structure
2. All Python files must be in `backend/src/`
3. Maintain single virtual environment setup
4. Test both frontend and backend integration

## Support

For issues:
1. Check logs in backend console
2. Verify file formats (JPG input only)
3. Ensure proper network connectivity between frontend/backend