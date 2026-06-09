# 🧠 Deepfake Detection System

A state-of-the-art deepfake detection system built with PyTorch and EfficientNet-B0, featuring a user-friendly web interface for real-time image and video analysis.

## 🌟 Features

- **Deep Learning Model**: EfficientNet-B0 architecture fine-tuned for deepfake detection
- **Multi-format Support**: Analyze both images (.jpg, .jpeg, .png) and videos (.mp4, .mov)
- **Web Interface**: Interactive Gradio-based web application for easy testing
- **Real-time Analysis**: Process first frame of videos for quick deepfake detection
- **Training Pipeline**: Complete PyTorch Lightning training infrastructure
- **Model Export**: Support for PyTorch (.pt) and ONNX format exports

## 📐 System Architecture

For detailed system architecture diagrams, data flow, and component interactions, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Usage

#### 🖥️ Web Application

Launch the interactive web interface:

```bash
python web-app.py
```

The web app will open in your browser where you can:
- Drag and drop images or videos
- View real-time predictions with confidence scores
- See preview of analyzed content


## 📂 Supported Datasets

This deepfake detection system supports various popular deepfake datasets. Below are the recommended datasets for training and evaluation:

### 🎬 Video-based Datasets

#### **FaceForensics++**
- **Description**: One of the most comprehensive deepfake datasets with 4 manipulation methods
- **Size**: ~1,000 original videos, ~4,000 manipulated videos
- **Manipulations**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Quality**: Raw, c23 (light compression), c40 (heavy compression)
- **Download**: [GitHub Repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Excellent for training robust models across different manipulation types

#### **Celeb-DF (v2)**
- **Description**: High-quality celebrity deepfake dataset
- **Size**: 590 real videos, 5,639 deepfake videos
- **Quality**: High-resolution with improved visual quality
- **Download**: [Official Website](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Usage**: Great for testing model performance on high-quality deepfakes

#### **DFDC (Deepfake Detection Challenge)**
- **Description**: Facebook's large-scale deepfake detection dataset
- **Size**: ~100,000 videos (real and fake)
- **Diversity**: Multiple actors, ethnicities, and ages
- **Download**: [Kaggle Competition](https://www.kaggle.com/c/deepfake-detection-challenge)
- **Usage**: Large-scale training and benchmarking

#### **DFD (Google's Deepfake Detection Dataset)**
- **Description**: Google/Jigsaw deepfake dataset
- **Size**: ~3,000 deepfake videos
- **Quality**: High-quality with various compression levels
- **Download**: [FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Additional training data for model robustness

### 🖼️ Image-based Datasets

#### **140k Real and Fake Faces**
- **Description**: Large collection of real and AI-generated face images
- **Size**: ~140,000 images
- **Source**: StyleGAN-generated faces vs real faces
- **Download**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)
- **Usage**: Perfect for image-based deepfake detection training

#### **CelebA-HQ**
- **Description**: High-quality celebrity face dataset
- **Size**: 30,000 high-resolution images
- **Quality**: 1024×1024 resolution
- **Download**: [GitHub Repository](https://github.com/tkarras/progressive_growing_of_gans)
- **Usage**: Real face examples for training


## 🛠️ Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Classifier**: Custom 2-class classifier with dropout (0.4)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake) with confidence scores

## 📊 Performance

- **Inference Speed**: Real-time on GPU, ~200ms per image on CPU
- **Input Support**: Images (.jpg, .png) and videos (.mp4, .mov)
- **Video Analysis**: 10-frame uniform sampling with probability averaging
- **Robustness**: Tested with Gaussian blur and JPEG compression noise simulation (`realeval.py`)

> **Note**: Accuracy metrics depend on your training dataset. Monitor `val_loss` and `val_acc` via TensorBoard during training.

