from flask import Flask, render_template, request
from joblib import load
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = load('devnagri.joblib')
label = load('labels.joblib')

# Function to preprocess image for prediction
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the model (assuming 32x32)
    resized_image = cv2.resize(gray_image, (32, 32))
    # Normalize the image pixel values
    preprocessed_image = resized_image.astype('float') / 255.0
    # Flatten the image to match the model input shape
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image

# Function to predict character from uploaded image
def predict_character(image_path):
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(image_path)

    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    prediction = prediction.argmax(axis=1)
    # Decode the predicted class (assuming your label encoder is saved along with the model)
    predicted_character = label.inverse_transform(prediction)

    return predicted_character[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Save the uploaded image temporarily
        uploaded_image_path = 'uploaded_image.png'
        image_file.save(uploaded_image_path)

        # Call the predict_character function to get the prediction
        predicted_character = predict_character(uploaded_image_path)

        # Delete the uploaded image file after processing
        os.remove(uploaded_image_path)

        return render_template('result.html', prediction=predicted_character)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
