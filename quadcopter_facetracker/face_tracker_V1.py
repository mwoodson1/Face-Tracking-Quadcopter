#!/usr/bin/env python

'''
Python code that creates and displays the video coming from the drone. Text is added to the image 
to provide relevant information about the current state of the drone and whether or not a face is 
detected/recognized. 

John Kirby

'''

HAAR_FILE = 'haarcascade_frontalface_default.xml'

import cv2
import cv
import numpy as np
import time 
from recogTrainer import gimlabs
from recogTrainer import faceRec


# For OpenCV image display
WINDOW_NAME = 'FaceTracker' 

# Path to image set 
path = '/home/facetrackers/Desktop/ARDroneAutoPylot-master/kksfaces'

# Path to caputured image
path2 = '/home/facetrackers/Desktop/ARDroneAutoPylot-master'

# Run the face trainer
gimlabs(path)

def track(img, battery, auto):
    '''
    '''

    # Assume no faces
    retval = None
    faceval = None

    # Set up state variables first time around
    if not hasattr(track, 'faceCascade'):

	track.conf = None
	track.name = None
	track.snap = 0        
	track.count = 0 
        track.start_time = time.time()
        track.faceCascade = cv2.CascadeClassifier(HAAR_FILE)


    track.count += 1

    # Convert the color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # scaleFactor: Compensates for different face sizes. In other tests a larger factor reduces false positives
    # minNeighbors: Defines how many objects are detected near the current one before it declares the face found
    # minSize: Gives the smallest dimensions that the detection will allow. Smaller = Farther away faces.

    faces = track.faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(15, 15),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Process only the first detected face for now
    if len(faces) > 0:

        x, y, w, h = faces[0]

	# Capture a frame from the video
	
	if w == 100 and auto and track.snap == 0:	
	    cv2.imwrite("subject.png", img)
	    track.snap += 1
	    print('Image Captured')
	    faceval = faceRec(path2)
	
	if not faceval is None:
	    track.name, track.conf = faceval
	    
	if not track.name is None:
	    cv2.putText(img, 'Identity: ' + track.name, (410,350), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
	    
	    
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Compute the center of the face rectangle
        ctr = (x+w/2, y+h/2)
	
	# Draw a little circle at the center
        cv2.circle(img, ctr, 5, (0, 0, 255), 2)
	
	# Display Fun Text
        cv2.putText(img, 'TARGET FOUND', (10,350), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
	if track.name is None:
	    cv2.putText(img, 'Identity: UNKNOWN', (410,350), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
	
        # Put center and width in return value
        retval = ctr, w


    # Remove name labels for next round and snap count for a new face
    if len(faces) == 0:
	track.name = None
	track.conf = None
	track.snap = 0

    # Display the battery percentage
    cv2.putText(img, 'Battery %d%%' % battery, (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,128,255), 2) 

    # Display the FPS
    cv2.putText(img, '%d' % int(track.count / (time.time() - track.start_time)), (610,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2) 

    # Displays a message if Autopilot is active
    if auto:
	cv2.putText(img, 'Autopilot ON', (10,45), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)

    # Resets the counter so another picture can be taken if autopilot is turned off
    if not auto:
	track.snap = 0
	track.conf = None
	track.name = None

    # Display full-color image
    cv2.imshow(WINDOW_NAME, img)
    
    # Force image display
    cv2.waitKey(1)

    return retval


