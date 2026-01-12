@echo off
echo ================================
echo Starting nnU-Net Training
echo Dataset: Dataset001_LungCancer
echo Configuration: 2D
echo ================================

nnUNetv2_train 1 2d 0 --npz
nnUNetv2_train 1 2d 1 --npz
nnUNetv2_train 1 2d 2 --npz
nnUNetv2_train 1 2d 3 --npz
nnUNetv2_train 1 2d 4 --npz

echo ================================
echo Training completed for all folds
echo ================================
pause
