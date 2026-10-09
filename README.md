# Image Preprocessing & Flower Classifier

An end-to-end Computer Vision project demonstrating image preprocessing pipelines, dataset batch streaming, and Convolutional Neural Network (CNN) classification built with Python and TensorFlow.

---
 
## 📌 Project Overview

Raw real-world image datasets are inherently messy: photos come in varying aspect ratios, disparate resolutions, and integer-encoded color values ranging between [0, 255]. Deep learning models require homogeneous tensor inputs and normalized numerical distributions to achieve stable gradient descent.

This project implements:
1. **Raw Image Inspection**: Analyzing dimensional variation and pixel value ranges directly from disk.
2. **Preprocessing Pipeline**: Resizing all images to a uniform 180x180x3 resolution and normalizing pixel values to [0.0, 1.0].
3. **CNN Architecture**: Building, compiling, and training a feature-extraction network.
4. **Live Inference**: Visualizing predictions against unseen validation samples with Softmax confidence scoring.

---

## 📊 Dataset Details 

The pipeline uses the **TensorFlow Flower Photos** dataset:
- **Total Images**: 3,670 color images
- **Classes (5 Categories)**: Daisy, Dandelion, Roses, Sunflowers, Tulips
- **Split Ratio**: 80% Training (2,936 images) / 20% Validation (734 images)

---

## 🔄 Image Preprocessing Pipeline

| Stage | Input Property | Processed Property | Purpose |
| :--- | :--- | :--- | :--- |
| **Resizing** | Arbitrary shapes (e.g., 263x320x3) | Standardized 180x180x3 | Guarantees fixed input tensor shape |
| **Normalization** | Integer values [0, 255] | Floating point [0.0, 1.0] | Stabilizes gradient backpropagation |
| **Batching** | Single unbatched files | Batches of 32 (`batch_size=32`) | Optimizes hardware memory cache |

---

## 🧠 Model Architecture

- **Input Layer**: (180, 180, 3)
- **Rescaling Layer**: Scales pixel values from [0, 255] to [0.0, 1.0]
- **Conv2D + ReLU**: 32 filters (3x3), same padding
- **MaxPooling2D**: (2x2) pool
- **Conv2D + ReLU**: 64 filters (3x3), same padding
- **MaxPooling2D**: (2x2) pool
- **Flatten**: Converts 2D feature maps to a 1D vector
- **Dense Layer**: 128 units, ReLU activation
- **Output Layer**: 5 units (Logits for the 5 flower classes)

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash 
   pip install -r requirements.txt 
