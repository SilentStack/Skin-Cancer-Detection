import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load data
data = []
labels = []

data_dir = "./images"  # Path to your images folder
categories = ["HAM10000_images_part_1", "HAM10000_images_part_2"]  # Subfolders containing the images

for category in categories:
    class_num = categories.index(category)  # Class index (0 for part_1, 1 for part_2)
    category_path = os.path.join(data_dir, category)
    
    # Ensure the folder exists
    if not os.path.exists(category_path):
        print(f"Warning: Path {category_path} does not exist!")
        continue
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            data.append(img)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Verify loaded data
if data.size == 0 or labels.size == 0:
    raise ValueError("No data or labels loaded. Check dataset path or structure.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize pixel values to range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encode the labels (for multi-class classification)
num_classes = len(categories)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Define the Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("skin_cancer_detector.h5")
print("Model training complete and saved as 'skin_cancer_detector.h5'.")
