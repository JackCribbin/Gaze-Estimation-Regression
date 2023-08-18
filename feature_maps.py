# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 18:25:35 2023

@author: jackp
"""

import matplotlib.pyplot as plt
import pathlib
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import cv2

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Change to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

#model = keras.models.load_model('models//tmp/xmodel_1.3_checkpoint')
#model = keras.models.load_model('models/tmp/ymodel_1.3_checkpoint')  
model = keras.models.load_model('models/tmp/ymodel_2_0.2_checkpoint')  



for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)
    
model = keras.Model(inputs=model.inputs , outputs=model.layers[6].output) 

print('\n')
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)
    
image_path = os.path.join(directory, 'example_eyes.jpg')
image = cv2.imread(image_path)   

# Convert the image into a keras tensor array
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
  
# Calculating features_map
features = model.predict(img_array)

print('features:',features.shape)

fig = plt.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
    #if(i == 26):    
     #   plt.figure()
      #  plt.imshow(features[0,:,:,i-1] , cmap='gray')
       # break
    #plt.show()
    
plt.show()



'''
class_namesLR = ['Left', 'MiddleH', 'Right']
class_namesUD = ['Down', 'MiddleV', 'Up']

# Make a prediction for the label of the image
# in regards to its horizontal positioning
predictions = model.predict(img_array)

# Compute the confidence level of the prediction
score = tf.nn.softmax(predictions[0])
print(score.shape)
print(np.argmax(score))
'''

'''
# Print the prediction and the confidence level
print(
    "\nThis image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_namesLR[np.argmax(score)], 100 * np.max(score))
)
'''

'''
# Convert the score Tensor to an array
scoreLR = np.array(score)

print(scoreLR)
'''

print('\nDone')







