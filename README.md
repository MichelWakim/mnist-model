# MNIST Model Trained in Google Colab

This repository contains a Python notebook that trains a convolutional neural network (CNN) on the MNIST dataset using Google Colab.

## Getting Started

To get started, you will need to have a Google account and access to Google Colab. You can open the notebook in Google Colab by clicking the "Open in Colab" button at the top of the notebook file in this repository.

## Dataset

The MNIST dataset is a collection of handwritten digit images that are commonly used for training and testing machine learning models. The dataset contains 60,000 training images and 10,000 test images.

## Model Architecture

The model used in this notebook is a simple CNN that consists of two convolutional layers, two max pooling layers, and two fully connected layers. The model achieves an accuracy of over 99% on the test set after training for 10 epochs.

## Results

The final accuracy achieved by the model on the test set is 98.60%.

## Usage

To use the trained model, you can load the model weights from the saved file and use it to make predictions on new images.

## Dependencies

This notebook requires the following Python libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib

## Acknowledgments

This notebook is based on the [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/keras/classification) and the [Keras Documentation](https://keras.io/examples/vision/mnist_convnet/).
