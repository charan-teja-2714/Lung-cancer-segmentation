# Lung Cancer Segmentation using Custom U-Net

A complete backend application for binary lung cancer segmentation using deep learning.

## Overview

This project provides an end-to-end solution for segmenting lung cancer from CT images using a custom U-Net architecture. The system can identify cancerous regions in lung CT scans.

### Cancer Classes
- **Background** (Class 0): No cancer
- **Cancer** (Class 1): Any cancer type (ADC, LCC, SCC combined)

## Tech Stack

**Backend:**
- Python 3.9+
- PyTorch
- Custom U-Net architecture
- OpenCV, NumPy, Pillow
- FastAPI (for inference API)

## Project Structure

```
lung-cancer-segmentation/
├── backend/
│   ├── src/                    # All Python files
│   │   ├── train.py           # Training script
│   │   ├── evaluate.py        # Evaluation script
│   │   ├── dataset.py         # Dataset loader
│   │   ├── model.py           # U-Net model
│   │   ├── main.py            # FastAPI application
│   │   └── inference.py       # Inference pipeline
│   ├── data/
│   │   └── raw/               # Training data
│   │       ├── train/
│   │       │   ├── CT/        # CT images by class
│   │       │   └── MASK/      # Mask images by class
│   │       └── test/
│   │           ├── CT/
│   │           └── MASK/
│   ├── checkpoints/           # Saved models
│   └── requirements.txt
└── README.md
```

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Training

```bash
cd backend
python src/train.py
```

### 3. Evaluation

```bash
cd backend
python src/evaluate.py
```

### 4. Run API Server

```bash
cd backend/src
python main.py
```

Backend runs on: `http://localhost:8000`

## API Endpoints

### POST /predict
- **Input**: JPG image file
- **Output**: PNG image with segmentation overlay
- **Timeout**: 60 seconds

### GET /
- Health check endpoint

## Training Features

- **Binary Segmentation**: Combines all cancer types into single class
- **Data Augmentation**: Built-in augmentation pipeline
- **Checkpointing**: Automatic model saving and resuming
- **Mixed Loss**: BCE + Dice loss for better segmentation
- **GPU Support**: CUDA acceleration when available

## Data Format

Training data should be organized as:
```
backend/data/raw/
├── train/
│   ├── CT/
│   │   ├── ADC/     # Adenocarcinoma CT images
│   │   ├── LCC/     # Large Cell Carcinoma CT images
│   │   └── SCC/     # Squamous Cell Carcinoma CT images
│   └── MASK/
│       ├── ADC/     # Corresponding masks
│       ├── LCC/
│       └── SCC/
└── test/            # Same structure for test data
```

## Model Architecture

- **Custom U-Net**: Encoder-decoder with skip connections
- **Input**: 256x256 grayscale images
- **Output**: Binary segmentation masks
- **Loss**: BCE + Dice Loss
- **Optimizer**: Adam (lr=1e-3)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional)

## Troubleshooting

### Training Issues
- Ensure data is in correct directory structure
- Check CUDA availability for GPU training
- Verify image formats (PNG/JPG supported)

### Memory Issues
- Reduce batch size in train.py
- Use CPU training if GPU memory insufficient
- Close other applications to free RAM

## License

Research Use Only

## Contributing

1. Follow the existing code structure
2. All Python files must be in `backend/src/`
3. Test training pipeline before submitting
4. Document any new features

## Support

For issues:
1. Check training logs for errors
2. Verify data directory structure
3. Ensure proper Python environment setup