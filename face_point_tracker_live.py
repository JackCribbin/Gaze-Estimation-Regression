# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:26:53 2023

@author: jackp
"""

#USAGE: python facial_68_Landmark.py

import dlib,cv2
import numpy as np
import sys
import os
sys.path.append('C:\\Users\jackp\OneDrive\Documents\_Regression')
from facePoints import facePoints, drawPoints
from imutils import face_utils

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
  with open(fileName, 'w') as f:
    for p in faceLandmarks.parts():
      f.write("%s %s\n" %(int(p.x),int(p.y)))

  f.close()

# Move to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

# location of the model (path of the model).
Model_PATH = "shape_predictor_68_face_landmarks.dat"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# now from the dlib we are extracting the method get_frontal_face_detector()
# and assign that object result to frontalFaceDetector to detect face from the image with 
# the help of the 68_face_landmarks.dat model
frontalFaceDetector = dlib.get_frontal_face_detector()


# Now the dlib shape_predictor class will take model
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

check = True
# Each image is then iterated through
count = -1
while True:
    
    # Capture frame-by-frame
    ret, image = cap.read()
    key = cv2.waitKey(1) & 0xFF
    
    count += 1
    
    # The image is conerted to full RGB 
    imageRGB = image#cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # The face detector attempts to find any faces present in the image
    allFaces = frontalFaceDetector(imageRGB, 0)
    
    check = True
    # Loop is ended is no faces or more than one face is detected
    if(len(allFaces) == 0):
        print("No faces detected")
        check = False
        #break
    elif(len(allFaces) > 1):
        print("Multiple Faces Detected")
        check = False
        #break
    
    if check:
        # The area containing the detected face is stored as a rectangle object
        faceRectangleDlib = dlib.rectangle(int(allFaces[0].left()),int(allFaces[0].top()),
            int(allFaces[0].right()),int(allFaces[0].bottom()))
      
        # Landmarks are then assigned to the face bounded within the rectangle area
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
        
        # Prints the number of landmarks that were successfully assigned
        print("Total number of face landmarks detected:",len(detectedLandmarks.parts()))
      
        # The detectedLandmarks object is converted to a numpy array of x and y points
        points = face_utils.shape_to_np(detectedLandmarks)
        
        # The drawPoints function is called to add these points to the image
        image = drawPoints(image, detectedLandmarks)
        
        # The landmarks are then saved to an external text file 
        fileName = 'output/image' + str(count)+ ".txt"
        print("Landmarks are saved into ", fileName)
        writeFaceLandmarksToLocalFile(detectedLandmarks, fileName)
    
        cv2.imshow("Webcam", image) 
        if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed      
            break


#cv2.imshow("Face landmark result", image)

# Pause screen to wait key from user to see result
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Close all windows 
cv2.destroyAllWindows()
cap.release() 

print('\nDone')
