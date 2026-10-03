# Aerial Image Segmentation (MLP from Scratch)

Pixel-wise semantic segmentation of aerial imagery into five classes —
Building, Road, Tree, Vehicle, Grass — using a Multi-Layer Perceptron
implemented from scratch with NumPy and SciPy (no high-level ML frameworks).

## Overview

Each pixel is represented by an 18-dimensional feature vector: raw RGB and
HSV channel values, plus the local mean and standard deviation of each
channel over a 5×5 neighbourhood. A 3-layer MLP (18→64→32→5) is trained with
mini-batch SGD, He initialisation, ReLU activations, and a softmax output,
using cross-entropy loss and manually implemented backpropagation.

## Results

| Image            | Balanced Accuracy |
|-------------------|-------------------:|
| Testing image 1   | 0.7528             |
| Testing image 2   | 0.6637             |
| **Overall**       | **0.7082**         |

## Method

1. **Feature extraction** — RGB + HSV channels normalised to [0,1]; mean and
   std per channel over a 5×5 patch (18 features/pixel)
2. **Stratified sampling** — 10% of pixels sampled per class for balance
3. **Normalisation** — z-score normalisation across all features
4. **Model** — MLP (18→64→32→5), He-initialised weights, ReLU hidden layers,
   softmax output
5. **Training** — mini-batch SGD (batch size 1024), 60 epochs, cross-entropy
   loss
6. **Inference** — same feature pipeline applied to test images; prediction
   saved as a PNG class-label mask

## Usage

```bash
pip install opencv-python numpy scipy matplotlib
python image_segmentation.py
```

Expects `training_image.jpg`, `training_mask.png`, `testing_image1.jpg`,
`testing_image2.jpg` in the working directory. 

## Limitations

Trained on a single image, which limits generalisation to unseen lighting
and scene conditions, and to underrepresented classes.


