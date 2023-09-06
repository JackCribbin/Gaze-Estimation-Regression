# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:25:35 2023

@author: jackp
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import cv2

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Change to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

# Set the model name and layer number to use and the image to test on
model_name = 'models/tmp/dualmodel_6_checkpoint'
layer_number = 12
image_name = 'example_eyes.jpg'

# Load the model
model = keras.models.load_model(model_name)
  
# Print out the model architecture  
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)

# Change the model to be a single layer
model = keras.Model(inputs=model.inputs , outputs=model.layers[layer_number].output) 

# Print out the layer architecture
print('\n')
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)

# Load an image in to test with
image_path = os.path.join(directory, image_name)
image = cv2.imread(image_path)   

# Convert the image into a keras tensor array
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
  
# Calculate the feature maps
features = model.predict(img_array)

# Display all of the feature maps
fig = plt.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
plt.show()

'''
# Display a single/multiple specific feature maps
fig = plt.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):
    if(i == 26):    
        plt.figure()
        plt.imshow(features[0,:,:,i-1] , cmap='gray')
        break
    plt.show()
'''   

print('\nDone')







