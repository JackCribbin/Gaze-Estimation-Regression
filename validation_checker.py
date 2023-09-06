# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:59:50 2023

@author: jackp
"""

import PIL
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the two models for horizontal and vertical position approximation
modelx = keras.models.load_model('models/tmp/xmodel_5_checkpoint')
modely = keras.models.load_model('models/tmp/ymodel_5_checkpoint')
modeldual = keras.models.load_model('models/tmp/dualmodel_6_short_checkpoint')

# Load in training images from folder
images = glob.glob('TestingImages/sample*jpg')
image_count = len(images)
print(image_count,'images found')

# Convert each of the images loaded into arrays
ims = {}
for im in images:
    ims[im]=np.array(PIL.Image.open(im))
test_ds = np.array([each for each in ims.values()]).astype(np.float32)

# Extract the pixel x and y values from the image filenames
test_xs = np.array([int(each[each.index('x=')+2 : each.index('y=')-1]) 
               for each in ims]).astype(np.float32)
test_ys = np.array([int(each[each.index('y=')+2 : each.index('.')]) 
               for each in ims]).astype(np.float32)

# Use the two models to predice x and y pixel values for each image
test_answersx = modelx.predict(test_ds) * 1980
test_answersy = modely.predict(test_ds) * 1080
test_answersdual = modeldual.predict(test_ds)

for i in range(len(test_answersdual)):
    test_answersdual[i][0] = test_answersdual[i][0]*1980
    test_answersdual[i][1] = test_answersdual[i][1]*1080

print(test_answersx[0])
print(test_answersy[0])
print(test_answersdual[0])

# Calculate what the maximum difference between the actual and predicted
# x pixel values was
maxx = 0
for i in range(test_answersx.size):
    
    dif = abs(test_xs[i]-test_answersx[i][0])
    if(dif > maxx):
        maxx = dif
        print('\nNew max x error:',str(dif))
        print(str(int(test_xs[i]))+' vs '+str(int(test_answersx[i][0])))    

print('\n\n\n')


# Calculate what the maximum difference between the actual and predicted
# y pixel values was
maxy = 0
for i in range(test_answersy.size):
    
    dif = abs(test_ys[i]-test_answersy[i][0])
    if(dif > maxy):
        maxy = dif
        print('\nNew max y error:',str(dif))
        print(str(int(test_ys[i]))+' vs '+str(int(test_answersy[i][0])))
        
print('\n\n\n')

print(test_answersdual.size)

# Calculate what the maximum difference between the actual and predicted
# y pixel values was
dualmaxx = 0
for i in range(int(test_answersdual.size/2)):
    
    
    difx = abs(test_xs[i]-test_answersdual[i][0])
    if(difx > dualmaxx):
        dualmaxx = difx
        print('\nNew max dual x error:',str(difx))
        print(str(int(test_xs[i]))+' vs '+str(int(test_answersdual[i][0])))
        
dualmaxy = 0

for i in range(int(test_answersdual.size/2)):
        
    dify = abs(test_ys[i]-test_answersdual[i][1])
    if(dify > dualmaxy):
        dualmaxy = dify
        print('\nNew max dual y error:',str(dify))
        print(str(int(test_ys[i]))+' vs '+str(int(test_answersdual[i][1])))
    

# Print the two values for maximum error on the predictions
print("\n\nMax x error:",maxx)
print("Max y error:",maxy)
print("\n\nMax dual x error:",dualmaxx)
print("Max dual y error:",dualmaxy)



print('\n\ndone')












