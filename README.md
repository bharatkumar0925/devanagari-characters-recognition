# Devanagari Characters Recognition

This project is aimed at developing a Devanagari characters recognition system using Convolutional Neural Networks (CNN). The system is designed to recognize handwritten characters in the Devanagari script commonly used in languages like Hindi, Nepali, Marathi, and others.

## Project Overview

The project includes the following components:

1. **CNN Model**: Utilizes TensorFlow and Keras to build a deep learning model for character recognition.

2. **Web Interface**: A Flask web application that allows users to upload images containing Devanagari characters for recognition.

3. **Pretrained Model**: Includes a pre-trained CNN model (`devnagri.joblib`) and label encoder (`labels.joblib`) for easy deployment.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Flask
- OpenCV
- Joblib
- Pandas
- Numpy

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/bharatkumar0925/devanagari-characters-recognition.git
   cd devanagari-characters-recognition
