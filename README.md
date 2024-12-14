# Authors
### Najib Haidar and Devan Perkash

# Link to Best Model
### https://drive.google.com/file/d/15iJsLreuKHEcDSiR9jQifn4FLY0grKVL/view?usp=sharing

# Project Overview

### The Emotion Classifier project aims to develop a neural network that can accurately classify human emotions based on an image of only the person's eyes. The primary goal is to create an end-to-end pipeline for training, validating, and testing a robust emotion classification model. This model has applications in human-computer interaction, affective computing, and mental health monitoring. The project includes well-structured modular code, a clear training and evaluation strategy, and a simple demo script to showcase its functionality. By leveraging convolutional neural networks (CNNs) for feature extraction and Transformer encoders for global context understanding, the model is designed to provide accurate emotion classification across multiple classes.

# Setup instructions

### To set up the environment and run this project on your local machine, follow these steps:
### 1. Clone the repository
### 2. Create a virtual environment (optional but recommended)
### 3. Install all required dependencies to the virtual environment. You can use:
### pip install -r requirements.txt

# How To Run

## Download the demo dataset from Google Drive:
### https://drive.google.com/drive/folders/1eQtgqIGsJnYhTgaM9stB51WdngOUaBsK?usp=drive_link

## Paste those 7 folders into the demo folder:

```
demo/
  ├── demo_data/
  ├── demo.py
```
### You can run the demo script using the following command:

### python demo.py

# Expected Output

### After running the demo script, you should see:
###	- Predictions of the model for the input test images.
###	- The classified emotion for each test image.

### The project is expected to classify human emotions into 7 distinct classes. When running the demo, you should see a printout of the following information:
###	-	Emotion Class: The predicted emotion for each test image (e.g., “Happy”, “Sad”, “Angry”).
###	-	Accuracy: The percentage of correctly predicted images from the test set.
###	-	Precision, Recall, and F1 Score: These metrics indicate how well the model is performing in terms of precision (positive predictive value), recall (sensitivity), and the balance between the two (F1 score).

# Model Architecture Information

### The Emotion Classifier is a hybrid model that combines the following key components:
###	1.	Convolutional Neural Network (CNN):
### Extracts low-level spatial features from input images.
###	2.	Transformer Encoder:
### Captures the global context of the input, learning relationships between features at different spatial locations.
###	3.	Fully Connected Head:
### Uses the extracted features to classify the emotion into one of the 7 target emotion classes.

### This architecture is designed to leverage both local spatial features (via CNNs) and global relationships (via Transformer encoders) for robust emotion classification.

# Training and Hyperparameter Details

### The key hyperparameters and settings include:
###	-	Number of Epochs: 20
###	-	Learning Rate: 0.001
###	-	Early Stopping Patience: 15 epochs without improvement in validation loss
###	-	Loss Function: Cross-Entropy Loss (suitable for classification tasks)
###	-	Optimizer: Adam optimizer

# Training Process

###	1.	Train the model on the training set for 20 epochs.
###	2.	After each epoch, validate the model on the validation set.
###	3.	If the validation loss improves, save the model weights.
###	4.	If validation loss does not improve for 15 consecutive epochs, stop training (early stopping).
# EyeEmotionClassifier
