# *Skin-Cancer-Classification*
<img src="skin cancer Image" alt="Image description" width="800" height="500">>
# 📑Project Title
### DeepSkinCancer: Detection and Classification Using Advanced Deep Learning Algorithms
# 📌Description
DeepSkinCancer: Detection and Classification Using Advanced Deep Learning Algorithms is a deep learning-based project aimed at early detection and classification of skin cancer using advanced Convolutional Neural Networks (CNNs). By leveraging pre-trained models like VGG16, ResNet50, DenseNet169, DenseNet201, and Xception, this project improves diagnostic accuracy in identifying nine different types of skin cancer from medical images.

The project utilizes data augmentation, preprocessing techniques, and fine-tuning of CNN architectures to achieve high classification accuracy. With a dataset of 2080 images from ISIC, the model is trained using SGD optimizer and categorical crossentropy loss function, evaluating performance based on accuracy, precision, recall, and F1-score.

The final trained model, which achieved the best results with Xception, is intended for deployment on AWS Cloud using Flask or FastAPI, making it accessible for real-world applications in dermatology.
# 📌Table of Contents
- [Project Overview](#project-0verview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Required Imports and Libraries](#Required-mports-and-Libraries)
- [Project Structure](#Project-Structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures Used](#Model-Architectures-Used)
- [Training Details](#training-details)
- [Performance Metrics](#Performance-Metrics)
## 📌Project Overview
This project aims to classify different types of skin cancer using deep learning models. By leveraging advanced CNN architectures, we improve diagnostic accuracy and provide an AI-driven solution for early skin cancer detection.
## 📂Dataset
Dataset Source: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

Total Images: 2080

Image Size: 224x224 pixels
## 🛠️Dependencies
Before running the project, ensure the following Python libraries are installed:

- `tensorflow`: For deep learning model development.

- `keras`: High-level API for building deep learning models.

- `matplotlib`: For data visualization.

- `numpy`: For handling numerical data.

- `pandas`: For data manipulation and analysis.

- `os`: For handling file operations.

To install these dependencies, run the following command:

```sh
pip install tensorflow keras matplotlib numpy pandas os
```
## 📌Required Imports & Libraries
```sh
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet169, DenseNet201, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
```
## 📁Project Structure
```sh
DeepSkinCancer/
│── dataset/               # Contains the dataset images
│── models/                # Pre-trained models and trained weights
│── src/
│   │── import_libraries.py  # Importing required libraries
│   │── import_dataset.py    # Loading and handling dataset
│   │── preprocess.py        # Data preprocessing and augmentation
│   │── image_processing.py  # Image transformations
│   │── train.py             # Model training (ResNet50, DenseNet169, VGG16, Xception, DenseNet201)
│   │── user_signup.py       # User sign-up implementation
│   │── user_signin.py       # User sign-in implementation
│   │── user_input.py        # Handling user inputs
│   │── final_outcome.py     # Producing the final classification outcome
│── requirements.txt       # Dependencies for the project
│── README.md              # Project documentation
│── app.py                 # Flask/FastAPI for deployment (optional)
```
## 🔄Data Preprocessing
### Training Data Augmentation:
```sh
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=False
)
```
### Test Data Preprocessing:
```sh
test_datagen = ImageDataGenerator(rescale=1./255)
```
## 🏗️Model Architectures Used
We experimented with multiple deep learning models:

- `VGG16`: A widely used Convolutional Neural Network (CNN) architecture with 16 layers. It follows a simple and uniform design with small 3×3 convolution filters and is known for its high accuracy in image classification tasks.

- `ResNet50`: A 50-layer deep residual network that introduces skip connections (residual learning) to avoid the vanishing gradient problem. It allows training of very deep networks with improved performance.

- `DenseNet169`: A densely connected CNN with 169 layers, where each layer receives input from all previous layers. This promotes feature reuse and reduces the number of parameters compared to traditional deep networks.

- `DenseNet201`: A deeper variant of DenseNet with 201 layers, further improving accuracy and efficiency through better feature propagation and gradient flow.

- `Xception`: An extension of the Inception architecture that uses depthwise separable convolutions, significantly reducing computational cost while maintaining high performance. It achieved the best accuracy in our experiments.
## 🎯Training Details

- Batch Size: 32

- Loss Function: Categorical Crossentropy

- Optimizer: Stochastic Gradient Descent (SGD)

- Number of Epochs: 20

- Evaluation Metrics: Accuracy, F1-Score, Precision, Recall
  
## 📊Performance Metrics

- Best Model: Xception

- Metrics Used: Accuracy, Precision, Recall, F1-Score

- Plots: Yes, training and validation accuracy/loss curves are included.
  
  `Loss`
  <img src="Xception Plots.png" alt="Image description" width="800" height="400">>
  
