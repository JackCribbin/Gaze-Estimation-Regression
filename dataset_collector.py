# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 02:46:13 2023

@author: jackp

This file uses the computer's webcam to collect images of a user's face.
It then finds and cuts out the sections of the image that contain their eyes,
stitch these together, and save this final image in a labelled folder
"""

import os
import pygame
import numpy as np
import cv2
import sys
import pyautogui

def findEyes(frame):
    '''
    Function to determine the location of a pair of eyes in a given image

    Parameters
    ----------
    frame : 3D float array
        The 480 x 640 x 3 color image taken from the webcam

    Returns
    -------
    eyes : 2D integer array
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

def makeScreen(check, x = -100, y = -100):
    '''
    Function to create a pygame screen to help with dataset collection

    Returns
    -------
    None.

    '''

    # Set the background color to white
    screen.fill((255, 255, 255))
    
    # Set the size of the red dot
    dot_size = 100
    
    # Set the color of the red dot 
    dot_color = (255, 0, 0)  # red
    
    #if(x != -1 and y != -1):
    if(check == False):
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
    elif(check == True):
        # Draw the red dot on the screen
        pygame.draw.circle(screen, (0,255,0), (x, y), dot_size)
            
    # Update the display
    pygame.display.update()

def cutoutEyes(frame, eyes):
    '''
    Function to cut-out and combine two images of the eyes found in an image

    Parameters
    ----------
    frame : 3D float array
        The 480 x 640 x 3 color image taken from the webcam
    eyes : 2D int array
        The array of coordinates for the 2 eyes, if they were found 

    Returns
    -------
    image : 3D float array
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
    
# Move to the directory where the file is located 
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
print('\nIn:',directory)

# Initialize pygame 
pygame.init()

# Set the display size
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

# Initialize webcam 
cap = cv2.VideoCapture(0)

# Set the file name number to start and end at
count = 1000
stop = 1500
slowdown = 0

#20/08/23: x model ok for now, collct more images 
#looking at top half of screen and retrain y model

collect = False
check = True
mousex = -100
mousey = -100

try:
    # Loop until the loop is broken from within
    found = False
    loop = True
    while loop:
        
        # Display the pygame screen
        makeScreen(check, mousex, mousey)
        
        if(collect == False):
            # Wait for user to close the window
            running = True
            while running:
                for event in pygame.event.get():
                    
                    # End the program if the window is quit
                    if event.type == pygame.QUIT:
                        running = False
                        loop = False
                    
                    # Check for when a key is pressed
                    if event.type == pygame.KEYDOWN:
                        
                        # If q is pressed, save the current frame from 
                        # the camera and end the loop                
                        if event.key == pygame.K_q:
                            collect = True
                            running = False
                        
                        # If escape is pressed, end the loop and the program
                        if event.key == pygame.K_ESCAPE:
                            loop = False
                            running = False
        slowdown = slowdown + 1
        # If the program hasn't been ended
        if(loop and slowdown % 5 == 0):
            print(slowdown)
            
            ret, frame = cap.read()
            mousex, mousey = pyautogui.position()
            
            # Call the findEyes() function on the image to find the 
            # location of eyes in the image
            eyes = findEyes(frame)
            
            
            
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
                elif(eyes[0][1] - eyes[1][1] > 100 or eyes[0][1] - eyes[1][1] < -100):
                    check = False
                    print('\nPlease level your camera or head out')
                
                # Checks if the user's eyes are too low
                elif(frame.shape[0] - eyes[0][1]  <= 100):
                    print('\nPlease move camera lower or head higher')
                    check = False
                # Checks if the user's eyes are too high
                elif(frame.shape[0] - eyes[1][1]  <= 100):
                    print('\nPlease move camera lower or head higher')
                    check = False
                
                    
                # If the user's eyes are in roughly the correct position
                if(check):
                    
                    # Display the image collected
                    #cv2.imshow('Eyes found',frame) 
                    
                    # Call the saveEyes() function to cut-out, combine and save an 
                    # image of both eyes
                    image = cutoutEyes(frame, eyes)
                    
                    # Specify the path to save the image to and save it
                    path = "TrainingImages/sample" + str(count) + "_x="+str(mousex)+"_y="+str(mousey)+".jpg"
                    cv2.imwrite(path, image)
                    
                    count = count + 1
                            
                 
            else:
                check = False
                print('No eyes found!')
            
            cv2.destroyAllWindows()
            
        # If the required number of images have been collected, end the program
        if(count == stop):
            loop = False
    
    # Destroy the webcam
    pygame.quit()
    cv2.destroyAllWindows()
    cap.release() 


# If an of some kind occurs, close all windows and inform the user
except Exception as e:
    print('###################################')
    print('\n\nThere was an error\n\n')
    print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
    print(repr(e))
    print('###################################')
    
    
    # Destroy the webcam
    cv2.destroyAllWindows()
    cap.release() 
    pygame.quit()
    
    
print('\nDone')


