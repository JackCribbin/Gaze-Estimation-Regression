# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:08:02 2023

@author: jackp

This file uses the computer's webcam to collect live images of a user's face.
It then loads a pretrained CNN model and estimates what section of the screen
the user is looking in, and displays this guess to the screen
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pygame
import sys
import cv2

def findEyes(frame):
    '''
    Function to determine the location of a pair of eyes in a given image

    Parameters
    ----------
    frame : 3D uint8 array
        The 480 x 640 x 3 color image taken from the webcam

    Returns
    -------
    eyes : 2D int32 array
        The array of coordinates for the 2 eyes, if they were found 

    '''
    # Convert the frame to grayscale for the haar classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the first haar classifier
    directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, 'haarcascade_eye_tree_eyeglasses.xml')
    eye_detector=cv2.CascadeClassifier(path)
    
    # Try to detect eyes in the frame
    eyes = eye_detector.detectMultiScale(gray,
            scaleFactor=1.05,
            minNeighbors=12,
            minSize=(80,80),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    
    # If the first eye detector didn't find both eyes
    if(len(eyes) != 2):
    
        # Set the second haar classifier
        path = os.path.join(directory, 'haarcascade_frontalface_alt.xml')
        eye_detector=cv2.CascadeClassifier(path)
        
        # Try to detect eyes in the frame
        eyes = eye_detector.detectMultiScale(gray,
                scaleFactor=1.05,
                minNeighbors=12,
                minSize=(80,80),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
    
    # If the first and second eye detector didn't find both eyes
    if(len(eyes) != 2):
    
        # Set the third haar classifier
        path = os.path.join(directory, 'haarcascade_eye.xml')
        eye_detector=cv2.CascadeClassifier(path)
        
        # Try to detect eyes in the frame
        eyes = eye_detector.detectMultiScale(gray,
                scaleFactor=1.05,
                minNeighbors=12,
                minSize=(80,80),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
    
    return eyes

def cutoutEyes(frame, eyes):
    '''
    Function to cut-out and combine two images of the eyes found in an image

    Parameters
    ----------
    frame : 3D uint8 array
        The 480 x 640 x 3 color image taken from the webcam
    eyes : 2D int32 array
        The array of coordinates for the 2 eyes, if they were found 

    Returns
    -------
    image : 3D uint8 array
        The 200 x 100 x 3 color image of both eyes

    '''
    # Make the eye cut-outs square, 100 x 100 pixels
    for i in range(2):
        for j in range(2,4):
            eyes[i][j] = 100
    
    # Convert the dimensions of the first eye cut-out
    # to coordinates for the square
    x1 = eyes[0][0]
    y1 = eyes[0][1]
    x2 = eyes[0][2] + x1
    y2 = eyes[0][3] + y1
    
    # Cut the eye section out of the frame
    roi1 = frame[y1:y2, x1:x2]
    
    # Convert the dimensions of the second eye cut-out
    # to coordinates for the square 
    x1 = eyes[1][0]
    y1 = eyes[1][1]
    x2 = eyes[1][2] + x1
    y2 = eyes[1][3] + y1
    
    # Cut the eye section out of the frame
    roi2 = frame[y1:y2, x1:x2]
    
    # Combine the two cutouts into one image
    if(eyes[0][0] > eyes[1][0]):
        image = np.concatenate((roi2,roi1),axis=1)
    else:
        image = np.concatenate((roi1,roi2),axis=1)
    
    return image
    
def displayGuess(x, y):
    '''
    Function to display a dot at where the model predicts the user 
    is looking

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
        
    # Fill the screen with white
    screen.fill((255, 255, 255))  # white
                    
    # Set the size of the red dot
    dot_size = 100
    
    # Set the color of the red dot 
    dot_color = (255, 0, 0)  # red
    
    # Fill the screen with white
    screen.fill((255, 255, 255))  # white
    
    # Display the webcam in the corner of the guess screen
    
    '''
    # Just eyes
    width = 600
    height = 300
    image1 = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    '''
    
    '''
    image1 = frame
    
    cutoff = 120
    for i in range(0,int(cutoff)):
        image1 = np.delete(image1,0,0)
        
    cv2.rectangle(image1,(eyes[0][0],eyes[0][1]-cutoff),
                  (eyes[0][0]+eyes[0][2],eyes[0][1]+eyes[0][3]-cutoff),(0,255,0),3)
    cv2.rectangle(image1,(eyes[1][0],eyes[1][1]-cutoff),
                  (eyes[1][0]+eyes[1][2],eyes[1][1]+eyes[1][3]-cutoff),(0,255,0),3)
    
    # Invert the frame
    display = np.rot90(image1)
    
    # The video uses BGR colors, convert to RGB for Pygame
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    
    # Add the frame to the surface
    surf = pygame.surfarray.make_surface(display)
    screen.blit(surf, (0,0))
    '''
    
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (x, y), dot_size)
    
    # Update the display
    pygame.display.flip()



# Turn off unneccesary error warnings
tf.get_logger().setLevel('ERROR')

# Move to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)


# Load the two models for horizontal and vertical position approximation
modelx = keras.models.load_model('models/tmp/xmodel_5_checkpoint')
modely = keras.models.load_model('models/tmp/ymodel_5_checkpoint')

# Initialize the webcam
cap = cv2.VideoCapture(0)
check = False

# Set the dimensions of the screen
size = [1920, 1080]

# Initialize Pygame
pygame.init()

# Create a Pygame display surface
screen = pygame.display.set_mode((size[0], size[1]), pygame.FULLSCREEN)

# Fill the screen with white
screen.fill((255, 255, 255))  # white


try:
    loop = True
    while loop:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF
        
        # Call the findEyes() function on the image to find the 
        # location of eyes in the image
        eyes = findEyes(frame)
        
        # If the user's eyes aren't in the correct position
        if(check == False):
            
            # Display the camera view with the region highlighted
            cv2.rectangle(frame,(100,340),(540,440),(0,255,0),3)
            #cv2.imshow('Eye View',frame) 
            
            mult = 2
            image1 = cv2.resize(frame, (frame.shape[1]*mult, frame.shape[0]*mult), 
                                interpolation = cv2.INTER_AREA)

            
            # Rotate the frame
            display = np.rot90(image1)
            
            # The video uses BGR colors, convert to RGB for Pygame
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            
        # If both eyes were found
        if(len(eyes) == 2):
            
            # Checks to ensure that the user's eyes are in the correct
            # region of view
            check = True
            
            # Checks if the user's eyes are too high
            if(eyes[0][1] <= 300):
                check = False
                print('\nPlease move camera higher or head lower')
            # Checks if the user's eyes are too low
            elif(eyes[0][1] >= 400):
                check = False
                print('\nPlease move camera lower or head higher')
            # Checks if the user's eyes are not level
            if(eyes[0][1] - eyes[1][1] > 100 or eyes[0][1] - eyes[1][1] < -100):
                check = False
                print('\nPlease level your camera or head out')
            # Checks if the bottom of the user's eyes are cut off
            elif(frame.shape[0] - eyes[0][1]  <= 100):
                print('\nPlease move camera lower or head higher')
                check = False
            # Checks if the top of the user's eyes are cut off
            elif(frame.shape[0] - eyes[1][1]  <= 100):
                print('\nPlease move camera lower or head higher')
                check = False
            
            
            # If the user's eyes are in roughly the correct position
            if(check):
                
                # Turn off the display of the webcam
                #cv2.destroyAllWindows()
                
                # Call the cutoutEyes() function to cut-out, combine and 
                # save an image of both eyes
                image = cutoutEyes(frame, eyes)
                
                # Convert the image into a keras tensor array
                img_array = tf.keras.utils.img_to_array(image)
                img_array = tf.expand_dims(img_array, 0)
                 
                # Make a prediction for the label of the image
                # in regards to its horizontal positioning
                predictionx = int(modelx.predict(img_array)*size[0])
                
                predictiony = int(modely.predict(img_array)*size[1])
                
                
                print(predictionx,',',predictiony)
                
                if(predictionx > size[0]):
                    predictionx = size[0]
                if(predictiony > size[1]):
                    predictiony = size[1]
                
                # Call the displayGuess() function to display a dot
                # on the screen that is an approximation of where the
                # user is looking
                displayGuess(predictionx, predictiony)
                
                                    
                # Checks for key press while the guess screen is up
                for event in pygame.event.get():
                    
                    # End the program if the window is quit
                    if event.type == pygame.QUIT:
                        running = False
                        loop = False
                    
                    # Check for when a key is pressed
                    if event.type == pygame.KEYDOWN:
                        # If escape is pressed, end the loop and the program
                        if event.key == pygame.K_ESCAPE:
                            loop = False
                            running = False
        
        # If no eyes are found, inform the user and repeat the loop
        else:
            check = False
            print('No eyes found!')
    
        # Checks for key press while the guess screen is up
        for event in pygame.event.get():
            
            # End the program if the window is quit
            if event.type == pygame.QUIT:
                running = False
                loop = False
            
            # Check for when a key is pressed
            if event.type == pygame.KEYDOWN:
                # If escape is pressed, end the loop and the program
                if event.key == pygame.K_ESCAPE:
                    loop = False
                    running = False
    
    # Destroy the webcam
    cv2.destroyAllWindows()
    cap.release() 
    
# If an of some kind occurs, close all windows and inform the user
except Exception as e:
    print('###################################')
    print('\n\nThere was an error\n\n')
    print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
    print(repr(e))
    print('###################################')
    
    cv2.destroyAllWindows()
    cap.release() 
    pygame.quit()

# Close all windows 
cv2.destroyAllWindows()
cap.release() 
pygame.quit()


print('\nDone')







