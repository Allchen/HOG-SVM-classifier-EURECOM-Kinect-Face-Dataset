## Synopsis

Simple Project for training a SVM with parameter tuning in Kinect Face Depth images of Eurecom Dataset, using HOG as a feature extractor.

All depth images have been cropped (using the keypoints provided by the databse) and resized 96x96 and normalized. 

Only neutral images are used for training, the remaining ones are kept for testing. 

## Current State

Dense grid performed between C and Gamma in a range of 10^-15 to 10^15. Saves the best model in a .XML file.

Note: OpenCV 3.1 with extra_modules is used (Boost lib is used to list all files in a directory).


