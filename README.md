# Instructions
To run any of the code you need to have OpenCV 3.1 installed. You can follow the instructions [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html) in order to do that.(The website says it will install 2.4 but the process will install 3.1) You may get an error when running the contour extraction or background subtraction part of the code. To fix this simply comment out the line that gives the error in the opencv library and make sure to recompile it.

# Gait Recognition
**classifier.py** will build 6 gait classifiers. Each classifier corresponds to a particular viewing angle of the person. These classifiers can be integrated with the face recognizing quad-copter code to enable tracking of persons even when the face is obstructed.

The classifier is implemented by the following steps:
- Perform background subtraction
- Extract the silhouette of the person
- Convert this silhouette to a 1D signal by computing the distance from the center of mass to the silhouette in a clockwise fashion
- Peform LDA paired with a KNN classifier. The hyperparameters are determined via grid search
