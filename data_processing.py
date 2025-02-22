import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from kagglehub import dataset_download  # Module to download datasets from Kaggle
from utils import load_image  # Custom function to load and preprocess images

# Function to download the dataset from Kaggle
def download_dataset():
    # Download dataset from Kaggle and store the path
    path = dataset_download("kmader/skin-cancer-mnist-ham10000")
    print(f"Dataset downloaded to: {path}")
    return path

# Function to get the path of each image file
def get_image_path(image_id, image_dir):
    # Loop through the subdirectories to locate the image file
    for subfolder in ['HAM10000_images_part_1', 'HAM_images_part_2']:
        # Construct full path to the image
        image_path = os.path.join(image_dir, subfolder, f"{image_id}.jpg")
        # Check if the image file exists at this path
        if os.path.exists(image_path):
            return image_path
    # Return None if the image is not found in either subfolder
    return None

# Function to preprocess the dataset
def preprocess_data():
    # Download dataset and retrieve the dataset path
    dataset_path = download_dataset()
    # Define the path to the metadata CSV file
    metadata_path = os.path.join(dataset_path, "HAM10000_metadata.csv")
    
    # Check if metadata file exists; raise error if not found
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata file 'HAM10000_metadata.csv' not found.")
    # Load metadata CSV into a pandas DataFrame
    metadata = pd.read_csv(metadata_path)

    # Initialize LabelEncoder for encoding disease labels as numerical values
    label_encoder = LabelEncoder()
    # Encode 'dx' column into numeric labels and store in 'label' column
    metadata['label'] = label_encoder.fit_transform(metadata['dx'])

    # Set the main image directory to the downloaded dataset path
    image_dir = dataset_path

    # Map each image_id to its corresponding file path
    metadata['image_path'] = metadata['image_id'].apply(lambda img_id: get_image_path(img_id, image_dir))
    # Drop rows where image paths could not be found (NaN values)
    metadata = metadata.dropna(subset=['image_path'])

    # If all images are missing, raise an error
    if metadata.empty:
        raise ValueError("No matching images found in the specified directories.")

    # Split data into training (80%) and test (20%) sets
    train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)

    # Load and preprocess training images from the file paths
    X_train = np.array([load_image(path) for path in train_data['image_path']])
    # Retrieve666 labels for the training set
    y_train = train_data['label'].values
    # Load and preprocess test images from the file paths
    X_test = np.array([load_image(path) for path in test_data['image_path']])
    # Retrieve labels for the test set
    y_test = test_data['label'].values

    # Return preprocessed training and test data and labels
    return X_train, X_test, y_train, y_test

# Main execution block
if __name__ == "__main__":
    try:
        # Call the preprocess function and store the returned data
        X_train, X_test, y_train, y_test = preprocess_data()
        # Print the size of the training and test sets
        print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    except Exception as e:
        # Print any errors that occur during preprocessing
        print("Error:", e)
