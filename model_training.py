import tensorflow as tf  # Importing TensorFlow library
from tensorflow.keras.models import Sequential  # Sequential model to build layers step by step
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Importing necessary layers for CNN

# Function to build the CNN model
def build_model():
    # Define a Sequential model
    model = Sequential([
        # First Conv2D layer with 32 filters, kernel size 3x3, and ReLU activation
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),  # Max pooling layer with a 2x2 pool size to reduce spatial dimensions

        # Second Conv2D layer with 64 filters, kernel size 3x3, and ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),  # Another max pooling layer to further reduce dimensions

        Flatten(),  # Flatten layer to convert 2D matrix to 1D vector for the Dense layer
        Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
        Dropout(0.5),  # Dropout layer to reduce overfitting by setting 50% of neurons to 0 during training

        Dense(7, activation='softmax')  # Output layer with 7 classes (for HAM10000) and softmax for class probabilities
    ])

    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model  # Return the compiled model

# Function to train the model
def train_model(X_train, y_train):
    model = build_model()  # Build the model architecture
    # Train the model on the training data with 10 epochs and 32 batch size, validating on 20% of the data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('skin_cancer_model.h5')  # Save the trained model to a file

# Main script to preprocess data and train the model
if __name__ == "__main__":
    from data_processing import preprocess_data  # Importing the preprocess function from data_preprocessing.py
    X_train, _, y_train, _ = preprocess_data()  # Load the training data and labels
    train_model(X_train, y_train)  # Train the model using the training data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")