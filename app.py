import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

# Load the trained model
model = load_model("skin_cancer_detector.h5")
print("Model loaded successfully.")

# Set up file upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image before prediction
def prepare_image(image_path):
    img = cv2.imread(image_path)  # Read the image from the specified file path
    img = cv2.resize(img, (128, 128))  # Resize the image to 128x128 pixels
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    image_path = None
    error = None

    if request.method == "POST":
        # Get patient data
        patient_id = request.form['patient-id']
        patient_name = request.form['patient-name']
        patient_age = request.form['patient-age']
        patient_gender = request.form['patient-gender']

        # Check if an image is uploaded
        if 'image' not in request.files:
            error = "No image file part"
            return render_template("skd.html", error=error)

        file = request.files['image']

        if file.filename == '':
            error = "No selected file"
            return render_template("skd.html", error=error)

        if file and allowed_file(file.filename):
            # Secure the filename and save the image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            image_path = url_for('static', filename='uploads/' + filename)  # Correct URL for static file
            
            # Prepare the image and predict
            img_array = prepare_image(filepath)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Mapping the prediction to a class name
            class_names = {
                0: "Actinic keratoses (AKIEC)",
                1: "Basal cell carcinoma (BCC)",
                2: "Benign keratosis-like lesions (BKL)",
                3: "Dermatofibroma (DF)",
                4: "Melanoma (MEL)",
                5: "Melanocytic nevi (NV)",
                6: "Vascular lesions (VASC)"
            }

            result = class_names.get(predicted_class, "Unknown")

    return render_template("skd.html", image_path=image_path, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
