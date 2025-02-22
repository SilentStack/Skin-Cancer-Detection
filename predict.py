import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2  # For image handling
from utils import load_image  # Assuming load_image function from data_processing.py can be reused

# Load the trained model from file
model = load_model("skin_cancer_detector.h5")
print("Model loaded successfully.")

# Function to load and preprocess the input image
def load_image(img_path):
    """
    Load and preprocess the image: resize, normalize, and expand dimensions.
    """
    img = cv2.imread(img_path)  # Read the image using OpenCV
    if img is None:
        print(f"Error: Unable to load image from {img_path}")
        return None
    
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class of an input image
def predict_image(img_path):
    # Dictionary to map label indices to skin condition names
    class_names = {
        0: "Actinic keratoses (AKIEC)",
        1: "Basal cell carcinoma (BCC)",
        2: "Benign keratosis-like lesions (BKL)",
        3: "Dermatofibroma (DF)",
        4: "Melanoma (MEL)",
        5: "Melanocytic nevi (NV)",
        6: "Vascular lesions (VASC)"
    }

    # Load and preprocess the input image
    img = load_image(img_path)  # Load the image using the custom load_image function
    if img is None:
        return "Error: Unable to load image."

    # Predict the class of the image using the model
    prediction = model.predict(img)
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Retrieve the class name from the dictionary
    predicted_class_name = class_names.get(predicted_class, "Unknown")
    return predicted_class_name

# Test the prediction function with a sample image
img_path = "abc.jpg"  # Replace with your test image path
predicted_class = predict_image(img_path)
print(f"Predicted class for the test image: {predicted_class}")
