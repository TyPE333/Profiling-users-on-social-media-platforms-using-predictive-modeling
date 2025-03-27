# -*- coding: utf-8 -*-
"""
Gender Classification based on Profile Images for MyPersonality Project
TCSS 551 Machine Learning
@author: Carla Peterson
Created on Sun Nov 19 09:40:21 2023
tcss555_cnnopencv.py

"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

# Check if two command-line arguments were provided
if len(sys.argv) != 3:
    print("Usage: mymodel.py <input1> <input2>")
    sys.exit(1)

# Retrieve the input values from command-line arguments
input_data_path= sys.argv[1]
output_data_path= sys.argv[2]

trainDirect = "/data/training/"
profileDirect = "profile/"
imageDirect = "image/"

# Load CSV data
csv_path = trainDirect + profileDirect + 'profile.csv'
df = pd.read_csv(csv_path)

# Load and preprocess images
image_dir = trainDirect + imageDirect
images = []
labels = []

for index, row in df.iterrows():
    img_path = os.path.join(image_dir, str(row['userid']) + '.jpg')
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize images to a consistent size
        images.append(img)
        labels.append(row['gender'])

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the CNN model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

print("CNN Model Summary")
print(model.summary())

'''
# Plotting (N/A on VM)
plot_model(model, to_file="cnnOpenCVImageModel.png", show_shapes=True, show_layer_activations=True, expand_nested=True)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt_title = "Sequential CNN Image Classifier for Gender\n3 CNN layers (32, 64 & 128 feature detectors), 2 Dense layers\nActivation: RELU & SOFTMAX, Optimizer: ADAM\nLoss: Binary Crossentropy, Batch Size: 64, Test Accuracy: " + str(round((test_acc * 100), 2)) + "%"

plt.title(plt_title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# End Plotting

# model.save("cnnOpenCVImageGender.keras")
'''

# This Section is for Testing the Model

# Load the profile data
test_profile_data=pd.read_csv(input_data_path + "profile/profile.csv",index_col=0)

tbl_dtypes = test_profile_data.dtypes
for k in range(0, len(tbl_dtypes)):
    # If the data type comes back as object, then convert it to string. Otherwise, errors will be thrown.
    if tbl_dtypes.iloc[k] == object:
        attr = tbl_dtypes.index[k]
        test_profile_data[attr] = test_profile_data[attr].convert_dtypes(infer_objects=True, convert_string=True)

# Create a filename column for easy access & referencing
test_profile_data['filename'] = test_profile_data['userid'] + '.jpg'

# Get a list of image file names in the directory
image_filenames = os.listdir(input_data_path + imageDirect)

# Make predictions for each image
predictions_list = []

for image_filename in image_filenames:
    # Load and preprocess the image
    image_path = os.path.join(input_data_path, imageDirect, image_filename)    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img, verbose=0)

    # Get the predicted class (0 for male, 1 for female)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Match the predicted class with the correct userid
    test_profile_data.loc[test_profile_data['filename'] == image_filename, ['gender']] = str(predicted_class)


# Print out the number of predictions for each class
num_male = test_profile_data.groupby('gender')['userid'].count().iloc[0]
num_female = test_profile_data.groupby('gender')['userid'].count().iloc[1]
print("Number of Female Predictions: " + str(num_female) + "\nNumber of Male Predictions: " + str(num_male))

# Output the xml files
for x in test_profile_data.userid:
    output_xml_file_name = output_data_path + str(x)+".xml"
    gender_value = test_profile_data.loc[test_profile_data['userid'] == x, 'gender'].values[0]
    if gender_value == '0':
        gender = "male"
    else:
        gender = "female"
        
    training_data_stats={
    "age_group": "xx-24",
    "gender": gender,
    "extrovert": "3.49",
    "neurotic": "2.73",
    "agreeable": "3.58",
    "conscientious": "3.45",
    "open": "3.91"

 	}
    temp={"id":str(x)}
    temp.update(training_data_stats)
    root = ET.Element("user",temp)
    tree = ET.ElementTree(root)
    tree.write(output_xml_file_name)
