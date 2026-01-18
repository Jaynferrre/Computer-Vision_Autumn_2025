# Computer-Vision Autumn 2025
## Project Overview

This repository contains the practical assignments and implementations associated with the Computer Vision Bootcamp. The primary objective of this project is to explore and implement deep learning architectures for image classification tasks using the PyTorch framework.

The coursework progresses from fundamental Artificial Neural Networks (ANN) to more advanced Convolutional Neural Networks (CNN), providing a comparative analysis of their performance on standard datasets. The project demonstrates the complete machine learning pipeline, including data preprocessing, model architecture design, training loops, evaluation metrics, and hyperparameter optimization.

## Repository Contents

The project is structured into two main assignments:

1.  **Assignment 1: Artificial Neural Networks (ANN)**
    * Focuses on the fundamentals of building a feedforward neural network.
    * Utilizes the MNIST dataset for handwritten digit classification.
2.  **Assignment 2: Convolutional Neural Networks (CNN)**
    * Focuses on architectures designed specifically for spatial data processing.
    * Utilizes the CIFAR-10 dataset for object classification.
    * Includes a comparative analysis between ANN and CNN models.
    * Implements advanced techniques such as data augmentation, regularization, and hyperparameter tuning.

## Technical Prerequisites

To execute the code contained in this repository, the following software and libraries are required:

* **Python 3.x**
* **PyTorch:** Core deep learning framework for tensor computation and neural networks.
* **Torchvision:** For dataset loading (MNIST, CIFAR-10) and image transformations.
* **Scikit-learn:** For performance metric calculations (Accuracy, Precision, Recall, F1-Score).
* **Matplotlib & Seaborn:** For data visualization and plotting confusion matrices.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Google Colab (Recommended):** For access to GPU acceleration (CUDA) to expedite training.

## Implementation Details

### Part 1: Artificial Neural Network (ANN) Implementation
This module establishes a baseline for image classification using a Multi-Layer Perceptron (MLP).

* **Data Preparation:** Implementation of `DataLoader` for the MNIST dataset with normalization transformations.
* **Architecture:** A fully connected network consisting of an input flattening layer, hidden linear layers with ReLU activation functions, and an output layer.
* **Training Regime:** Application of the Adam optimizer and Cross-Entropy Loss function over a defined number of epochs.
* **Evaluation:** Assessment of the model using a confusion matrix and classification report metrics.

### Part 2: Convolutional Neural Network (CNN) Implementation
This module addresses the limitations of ANNs in handling complex image data by preserving spatial hierarchies through convolution.

* **Data Preparation:** Loading and preprocessing the CIFAR-10 dataset.
* **Architecture:**
    * **Convolutional Layers:** To extract local features such as edges and textures.
    * **Pooling Layers:** To perform down-sampling and reduce dimensionality.
    * **Fully Connected Layers:** For final classification based on extracted features.
* **Performance Comparison:** A direct comparison illustrating the superior accuracy of CNNs over ANNs for complex image datasets.

### Advanced Techniques
The project extends beyond basic implementation to include optimization and analysis strategies:

* **Hyperparameter Tuning:** Systematic search logic to identify optimal learning rates, batch sizes, and epoch counts.
* **Regularization:** Implementation of Dropout layers to mitigate overfitting.
* **Data Augmentation:** Application of transformations such as `RandomHorizontalFlip` and `RandomCrop` to increase dataset diversity and improve model generalization.
* **Feature Visualization:** Visual inspection of convolutional filter weights to interpret feature extraction mechanisms.
* **Error Analysis:** Visualization of misclassified images to diagnose model weaknesses.

## Usage Instructions

1.  Ensure a Python environment is configured with the necessary dependencies.
2.  Open the notebooks (`24B2176_CV_Assignment_1.ipynb` or `24B2176_CV_Assignment_2.ipynb`) in Jupyter Notebook or Google Colab.
3.  Execute the cells sequentially. It is highly recommended to set the hardware accelerator to GPU if available to ensure efficient training times.
4.  The output cells will display training loss graphs, accuracy metrics, and confusion matrices.

## Metrics

The models are evaluated based on the following key performance indicators:

* **Accuracy:** The ratio of correctly predicted observations to the total observations.
* **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
* **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all observations in the actual class.
* **F1 Score:** The weighted average of Precision and Recall.

## Acknowledgments

This project was developed as part of a comprehensive curriculum covering PyTorch fundamentals, Neural Networks, Transfer Learning, and Object Detection.
