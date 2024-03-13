# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data processing
import tensorflow  # For deep learning framework
import keras  # For building neural networks
import os  # For interacting with the operating system
import cv2  # For image processing
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten  # Layers for building CNN
import random  # For generating random numbers
from sklearn.model_selection import train_test_split  # For splitting data into train and validation sets
from keras.models import Sequential  # Sequential model for stacking layers

# Function to create training data
def create_trainData(DataDirectory, img_size):
    train_data = []  # List to store training data
    Classes = [i for i in os.listdir(DataDirectory)]  # List of classes (emotions)
    for category in Classes:
        path = os.path.join(DataDirectory, category)  # Path to each class directory
        class_num = Classes.index(category)  # Assigning a numerical label to each class
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))  # Reading the image
            new_array = cv2.resize(img_array, (img_size, img_size), 3)  # Resizing the image
            new_array = new_array.reshape(img_size, img_size, 3)  # Reshaping to match model input
            train_data.append([new_array, class_num])  # Appending image and label to training data
    return train_data

# Creating training dataset
train_dataset = create_trainData('../input/fer2013/test', 299)  # Providing directory and image size

# Separating features (X) and labels (y)
X = []
y = []
for features, label in train_dataset:
    X.append(features)
    y.append(label)
X = np.array(X)
y = np.array(y)

# Splitting data into training and validation sets
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.1)

# Building the model architecture
model = tensorflow.keras.applications.Xception()  # Using Xception as base model
base_input = model.layers[0].input  # Input layer of the base model
base_output = model.layers[-2].output  # Output layer of the base model
final_output = keras.layers.Dropout(0.1)(base_output)  # Adding dropout layer
final_output = keras.layers.Dense(7, activation='softmax')(final_output)  # Output layer with softmax activation
new_model = keras.Model(inputs=base_input, outputs=final_output)  # Creating new model with modified layers

# Compiling the model
new_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss function
                  optimizer='adam',  # Optimizer
                  metrics=['accuracy'])  # Evaluation metric

# Training the model
new_model.fit(Xtrain, ytrain, epochs=20, validation_data=(Xval, yval))

# Saving the trained model
new_model.save('emotion.h5')
