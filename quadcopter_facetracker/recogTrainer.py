#!/usr/bin/env python

"""
Face Recognizer
This python code trains the recognizer using a provided image database. The recognizer is run when 
an image is pulled from the video feed. If the face is recognized with a sufficent confidence,
the name of the person is returned.

John Kirby

"""

import cv2, os
import numpy as np
from PIL import Image

# Load the facial recognition tool
HAAR_FILE = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(HAAR_FILE)

recognizer = cv2.createLBPHFaceRecognizer(neighbors = 4, grid_x = 8, grid_y = 8)

# Path to image set 
path = '/home/facetrackers/Desktop/ARDroneAutoPylot-master/kksfaces'

# Path to caputured image
path2 = '/home/facetrackers/Desktop/ARDroneAutoPylot-master'

def gimlabs(path):
	# Append all the absolute image paths in a list image_paths
	# Will not read the image with the .sad extension in the training set
	# Use .sad files for testing
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
	# images contain face images
	images = []
	# label has the label that is assigned to the image
	labels = []
	for image_path in image_paths:
		# Read and convert to grayscale
		image_pil = Image.open(image_path).convert('L')
		# Convert to numpy array
		image = np.array(image_pil, 'uint8')
		# Get image label
		nbr = int(os.path.split(image_path)[1].split('.')[0].replace('subject',''))
		# Detect face in image
		faces = faceCascade.detectMultiScale(image, minSize = (95,95))#, scaleFactor=1.3, minNeighbors=8, minSize=(100, 100), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		# If face detected, append face to images and label to labels 
		for (x, y, w, h) in faces:
			#images.append(image) 
			images.append(image[y: y+h, x: x+w])
			labels.append(nbr)
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			#cv2.imshow("Adding Faces to training set...", image) 
			cv2.imshow("Adding Faces to training set...", image[y: y+h, x: x+w])
			cv2.waitKey(1)
	#Return image list and labels and destroy face window
	cv2.destroyWindow("Adding Faces to training set...")	
	return images, labels
	
# Call the get images and labels function
images, labels = gimlabs(path)

# Perform the training
recognizer.train(images, np.array(labels))


# Perform the face recognition on the captured image
def faceRec(path2):
	#Grab the file
	image_paths = [os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.png')]
	for image_path in image_paths:
		# Read and convert the grayscale
		predict_image_pil = Image.open(image_path).convert('L')
		predict_image = np.array(predict_image_pil, 'uint8')
		# Detect the seen face
		faces = faceCascade.detectMultiScale(predict_image, minSize = (95,95))
		for (x, y, w, h) in faces:
			nbr_predicted, conf = recognizer.predict(predict_image[y: y+h, x: x+w])

			# print the recognition of who is seen in the terminal
			if not nbr_predicted is None and conf > 90:
				name = 'Results Inconclusive'
			elif nbr_predicted == 1:
				name = 'John'
				print 'Recognized as John with confidence: {}'.format(conf)
			elif nbr_predicted == 2:
				name = 'Brian'
				print 'Recognized as Brian with confidence: {}'.format(conf)
			elif nbr_predicted == 3:
				name = 'Joe'
				print 'Recognized as Joe with confidence: {}'.format(conf)
		
			# Capture the values of the face
			faceval = name, conf
			# Return the values to be shown on screen
			#if not name is None and int(conf) < 90:
			return faceval

			# Show the captured image
			#cv2.imshow('Recognizing Face', predict_image[y: y+h, x: x+w])
			#cv2.waitKey(5000)


