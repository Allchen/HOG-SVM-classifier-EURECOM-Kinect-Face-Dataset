#include "boost/filesystem.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/detection_based_tracker.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace boost::filesystem;
using namespace cv;
using namespace cv::ml;

vector<string> listAllFilesInDirectory(path filePath){

	vector<string> files;

	directory_iterator end_itr;

	// cycle through the directory
	for (directory_iterator itr(filePath); itr != end_itr; ++itr)
	{
		// If it's not a directory, list it. If you want to list directories too, just remove this check.
		if (is_regular_file(itr->path())) {
			// assign current file name to current_file and echo it out to the console.
			string current_file = itr->path().string();
			files.push_back(current_file);
		}
	}

	return files;
}


// Adapted for 12 test images and 52 identitys, if needed Change this
float getClassifierAccuracy(Mat predicted, Mat groundTruth){
	int correct_predictions = 0;
	for (int i = 0; i < 52 * 12; i++){
		if ((int) predicted.at<float>(i) == groundTruth.at<int>(i))
		{
			correct_predictions++;
		}
			
	}
	
	float accuracy = (float) correct_predictions / (52 * 12);
	return accuracy;
}


int main(){

	path DatabaseFilePath = "E:/Thesis Datasets/EURECOM - Preprocessed/KinectDB Normalize/";
	vector<string> files = listAllFilesInDirectory(DatabaseFilePath);

	Mat image;

	HOGDescriptor hog;
	vector<float> descriptors;

	int samples_visited = 0;
	
	// Each image will have a feature vector of 512 features. We need memory allocation due to the array size.
	Mat trainingDataMat(52 * 2, 512, CV_32F);
	Mat labelsMat(52 * 2, 1, CV_32S);
	

	cout << "Prepare for feature extraction ... " << endl << endl; 

	// Prepare Data. Only use depth neutral images for training. Images are 96 x 96.
	for (int i = 1; i < files.size() - 5;  i = i + 7){
		image = imread(files[i], 0); 
		hog = HOGDescriptor(cvSize(96, 96), cvSize(12, 12),	cvSize(12, 12), cvSize(12, 12), 8);
		labelsMat.at<int>(samples_visited) = (int) samples_visited / 2 + 1;
		// Get a 512 feature vector
		hog.compute(image, descriptors, Size(1, 1), Size(0, 0));

		// Save feature vector in the trainingData
		for (int j = 0; j < descriptors.size(); j++){
			trainingDataMat.at<float>(samples_visited, j) = descriptors[j];
		}

		
		cout << "Feature Extraction of Training image " << samples_visited + 1 << " of a total of " << 52 * 2 << endl;
		samples_visited++;
	}

	

	// Prepare test data
	Mat testDataMat(52 * 12, 512, CV_32F);
	Mat labelsTest(52 * 12, 1, CV_32S);
	string current_filename;

	samples_visited = 0;

	for (int i = 0; i < files.size(); i++){
		current_filename = files[i].substr(files[i].size()-6,2);
		if (current_filename != "al"){
			image = imread(files[i], 0);
			hog = HOGDescriptor(cvSize(96, 96), cvSize(12, 12), cvSize(12, 12), cvSize(12, 12), 8);
			labelsTest.at<int>(samples_visited) = (int) samples_visited / 12 + 1;
			// Get a 512 feature vector
			hog.compute(image, descriptors, Size(1, 1), Size(0, 0));

			// Save feature vector in the trainingData
			for (int j = 0; j < descriptors.size(); j++){
				testDataMat.at<float>(samples_visited, j) = descriptors[j];
			}

			cout << "Feature Extraction of Test image " << samples_visited + 1 << " of a total of " << 52 * 12 << endl;
			samples_visited++;
		}


	}

	cout << endl <<  "Feature Extraction ended successfully!" << endl << endl;
	cout << "Training C-SVM, using a RBF kernel with default parameters... " << endl;




	//// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->save("Model.xml");

	cout << "Train completed. Model has been saved as Model.xml" << endl << endl;

	cout << "Predicting the labels of test images..." << endl << endl;

	Mat predictedLabels(52 * 12, 1, CV_32S);
	svm->predict(testDataMat, predictedLabels);

	cout << "Prediction completed!" << endl << endl;

	cout << "Default Classifier Accuracy: " << getClassifierAccuracy(predictedLabels, labelsTest) * 100 << " % " << endl << endl;


	system("Pause");
	return 0;
}