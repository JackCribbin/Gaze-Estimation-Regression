# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:58:57 2023

@author: jackp
"""

import PIL
import glob
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pathlib
from sklearn.model_selection import train_test_split

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set the number of epochs to train the model for
epochs = 300

# Choose what type of model (horizontal/vertical) to train and what to call it
model_type = 'y'
model_name = 'model_2_0.2'

# Set where to save model checkpoints
checkpoint_filepath = 'c:\\users\\jackp\\onedrive\\documents\\_regression\\models\\tmp\\'+model_type+model_name+'_checkpoint'

# Load in training images from folder
images = glob.glob('TrainingImages/sample*jpg')
image_count = len(images)
print(image_count,'images found')

# Convert each of the images loaded into arrays
ims = {}
for im in images:
    ims[im]=np.array(PIL.Image.open(im))
train_ds = np.array([each for each in ims.values()]).astype(np.float32)

# Extract the pixel x and y values from the image filenames
xs = np.array([int(each[each.index('x=')+2 : each.index('y=')-1]) 
               for each in ims]).astype(np.float32)
ys = np.array([int(each[each.index('y=')+2 : each.index('.')]) 
               for each in ims]).astype(np.float32)

# Split the datasets and pixel values into training and validation datasets
if(model_type == 'x'):
    train_ds, val_ds, train_xs, val_xs = train_test_split(train_ds, xs, test_size=0.3, random_state=101)
elif(model_type == 'y'):
    train_ds, val_ds, train_ys, val_ys = train_test_split(train_ds, ys, test_size=0.3, random_state=101)

# Initialize a data augmenter for expanding the dataset
data_augmentation = keras.Sequential(
  [
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Input the image dimensions and number of color layers for any needed rescaling
img_height = 100
img_width = 200
num_layers = 3

# Initialize the model and all of its layers
model = tf.keras.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, num_layers)),
  data_augmentation,
  layers.Conv2D(16, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  tf.keras.layers.Dense(units=512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(units=256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(units=64, activation='relu'),
  tf.keras.layers.Dense(units=1)
])

# Compile the model and print a summary of its structure
model.compile(loss='mean_squared_error', optimizer="adam")
model.summary()

# Initialize a callback for saving checkpoint models whenever the validation
# loss decreases
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, monitor='val_loss', mode='min', 
    save_best_only=True)

# Initialize a callback for stopping the model training early if no improvement
# is seen after 30 epochs
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience = 50)

# Train the model
if(model_type == 'x'):
    history = model.fit(train_ds, train_xs, validation_data=(val_ds,val_xs), 
                        epochs=epochs, #batch_size=64, #verbose=1,
                        callbacks=[checkpoint_callback])#, early_stop_callback])
elif(model_type == 'y'):
    history = model.fit(train_ds, train_ys, validation_data=(val_ds,val_ys), 
                        epochs=epochs, #batch_size=64, #verbose=1,
                        callbacks=[checkpoint_callback, early_stop_callback])
                        
# Save the final model trained
model.save('models/'+model_type+model_name+'master')

# Save the training and validation loss for later plotting
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print what the minimums for loss and validation loss and at what epoch they
# occured
print('Minimum for loss was at epoch:',np.argmin(loss),'of',np.min(loss))
print('Minimum for validation loss was at epoch:',np.argmin(val_loss),'of',np.min(val_loss))

# Save the range of epochs for later plotting
epochs_range = range(len(loss))

# Plot the loss and validation loss over the epochs
plt.figure()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel("Epoch Number", fontsize=20)
plt.ylabel("Model Loss", fontsize=20)
plt.xlim(-1, epochs+1)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=18)
plt.show()

minloss = 999999999
minval = 999999999

# If more than 100 epochs occured, print what the minimum loss and validation
# loss were for that range
if(len(loss) > 100):
    for i in range(100):
        if(loss[i] < minloss):
            minloss = loss[i]
        if(val_loss[i] < minval):
            minval = val_loss[i]

    print('(100) Minimum for loss was at epoch:',loss.index(minloss),'of',minloss)
    print('(100) Minimum for validation loss was at epoch:',val_loss.index(minval),'of',minval)


minloss = 999999999
minval = 999999999

# If more than 200 epochs occured, print what the minimum loss and validation
# loss were for that range
if(len(loss) > 200):
    for i in range(100, 200):
        if(loss[i] < minloss):
            minloss = loss[i]
        if(val_loss[i] < minval):
            minval = val_loss[i]

    print('(200) Minimum for loss was at epoch:',loss.index(minloss),'of',minloss)
    print('(200) Minimum for validation loss was at epoch:',val_loss.index(minval),'of',minval)


'''
loss_short = []
val_loss_short = []

for i in range(50):
    loss_short[i] = loss[len(loss)-50+i]
    val_loss_short[i] = val_loss[len(val_loss)-50+i]
    
epochs_range_short = range(50)

# Plot the loss over the epochs
plt.figure()
plt.plot(epochs_range_short, loss_short, label='Training Loss')
plt.plot(epochs_range_short, val_loss_short, label='Validation Loss')
plt.xlabel("Epoch Number", fontsize=20)
plt.ylabel("Model Loss", fontsize=20)
plt.xlim(-1, epochs+1)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=18)
plt.show()
'''


########### Testing ####################

'''
test_ims = glob.glob('TestingImages/sample*jpg')
test_ims = {}
for png in test_ims:
    test_ims[png]=np.array(PIL.Image.open(png))

test_ds = np.array([each for each in test_ims.values()]).astype(np.float32)
test_xs = np.array([int(each[each.index('x=')+2 : each.index('y=')-1]) 
               for each in test_ims]).astype(np.float32)
test_ys = np.array([int(each[each.index('y=')+2 : each.index('.')]) 
               for each in test_ims]).astype(np.float32)

test_answers = model.predict(test_ds)

if(model_type == 'x'):
    print(test_xs)
    
elif(model_type == 'y'):
    print(test_ys)
print(test_answers)
'''

print('done')












