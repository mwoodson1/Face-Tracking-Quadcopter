# Instructions
To run any of the code you need to have OpenCV 3.1 installed. You can follow the instructions [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) in order to do that.(The website says it will install 2.4 but the process will install 3.1) You will also probably get an error when running the contour extraction or background subtraction. To fix this simply comment out the line that gives the error in the opencv library and make sure to recompile it.

**background_subtraction.py** will show how to perform background subtraction. 

**contour_extraction.py** will show how to get the contour around the people detected

**face_detection.py** you can run this by running *python face_detection.py haarcascade_frontalface_default.xml*. This will perform face detection on your webcam input.

**classifier.py** is currently under construction and is where the gait recognition system as a whole is being implemented.
