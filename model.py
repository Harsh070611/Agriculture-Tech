import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.api.layers import Flatten, Dense, Dropout
from keras.api.regularizers import l2

# Dataset path
dataset_path = os.path.join(os.getcwd(), "UTKFace")

# Initialize lists for images, ages, and genders
images = []
ages = []
genders = []

counter = 0
# Loop over all files in the dataset
for file in os.listdir(dataset_path):
    counter = counter+1
    if(counter >1500):
        break
    if file.endswith(".jpg") or file.endswith(".png"):
        # Extract age and gender from the filename
        age = int(file.split("_")[0])
        gender = int(file.split("_")[1])
        # Read the image
        img = cv2.imread(os.path.join(dataset_path, file))
        
        # Resize image to 224x224 (or as per CNN input requirement)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        
        # Append the image, age, and gender
        images.append(img)
        ages.append(age)
        genders.append(gender)

# Convert lists to numpy arrays
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

# Define age bins for classification
bins = [0, 10, 20, 35, 50, 70, 99, 116]  # Bins based on age groups: 0-10, 11-20, etc.
labels = range(len(bins) - 1)  # Labels will be [0, 1, 2, 3, 4, 5, 6]

# Use pandas to bin the ages
age_bins = pd.cut(ages, bins=bins, labels=labels, right=True)

# Reshape to 2D array because OneHotEncoder requires a 2D input
age_bins = age_bins.to_numpy().reshape(-1, 1).astype(int)  # Use astype(int) to ensure proper encoding

# Initialize OneHotEncoder for age bins
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform the age bins to one-hot encoded form
encoded_ages = encoder.fit_transform(age_bins)

# encoded_ages now contains the one-hot encoded age groups
print(encoded_ages)

# Split data into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    images, encoded_ages, test_size=0.1, shuffle=True, random_state=42
)

# Preprocess images for VGG16
features_train = keras.applications.vgg16.preprocess_input(features_train)
features_test = keras.applications.vgg16.preprocess_input(features_test)

# Add preprocessing layer at the front of VGG16
vgg = keras.applications.VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Prevent training already trained layers
for layer in vgg.layers:
    layer.trainable = False
  
# Add flatten layer
x = Flatten()(vgg.output)

# More Dense layers
# Use weight regularization (L2 vector norm) and dropout layers to reduce overfitting
x = Dense(1000, activation="relu", kernel_regularizer=l2(0.001))(x)


x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)


# Dense layer with number of neurons equal to number of classes
prediction = Dense(labels_train.shape[1], activation='softmax')(x)

# Create the model object
model = keras.Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

# Model summary
model.summary()

# Train the model
history = model.fit(features_train, labels_train, epochs=5, shuffle=True, validation_split=0.1)

# Evaluate the model
model_evaluation_history = model.evaluate(features_test, labels_test)


def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Resize the image to match your model's input (224x224)
    img = cv2.resize(img, (224, 224))
    
    # Normalize the image (since you normalized your training images between 0-1)
    img = img / 255.0
    
    # Expand dimensions to match the model input (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

# Path to the uploaded image
image_path = "harsh.jpg"  # Replace with the actual path to the image

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Load your trained model if not already loaded
# model = ... (load the trained model here if it's not loaded)

# Predict the age group using the model
prediction = model.predict(preprocessed_image)

# Get the predicted age group (it's a softmax output)
predicted_age_group = np.argmax(prediction, axis=1)

# Define the age bins (same as your training)
bins = [0, 10, 20, 35, 50, 70, 99, 116]  # These are the bins used during training

# Print the predicted age range
print(f"Predicted age group: {bins[predicted_age_group[0]]}-{bins[predicted_age_group[0]+1]}")