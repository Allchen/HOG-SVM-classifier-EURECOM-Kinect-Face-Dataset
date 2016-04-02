## Synopsis

Simple Project for training a SVM with parameter tuning in Kinect Face Depth images of Eurecom Dataset, using HOG as a feature extractor.

All depth images have been cropped (using the keypoints provided by the databse) and resized 96x96 and normalized. 

Only neutral images are used for training, the remaining ones are kept for testing. 

## Current State

Default classifier accuracy provided, no parameter tuning. Model is saved in a XML

Note: OpenCV 3.1 with extra_modules is used (Boost lib is used to list all files in a directory).


