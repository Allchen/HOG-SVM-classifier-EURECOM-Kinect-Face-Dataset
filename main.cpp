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
#include <math.h>

using namespace std;
using namespace boost::filesystem;
using namespace cv;
using namespace cv::ml;

// Outputs a vector of strings with all the files in a path.
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


// Adapted for 12 test images and 52 identitys, if needed Change this.
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

	// Get all images available in the dataset path.
	path DatabaseFilePath = "E:/Thesis Datasets/EURECOM - Preprocessed/KinectDB Normalize/";
	vector<string> files = listAllFilesInDirectory(DatabaseFilePath);

	// Variables initialization
	Mat image;
	// HOG descritor adapted for 96x96 images using 12x12 blocks, dividing the image in 8x8 blocks.
	HOGDescriptor hog = HOGDescriptor(cvSize(96, 96), cvSize(12, 12), cvSize(12, 12), cvSize(12, 12), 8); 
	vector<float> descriptors;
	
	// Each image will have a feature vector of 512 features. We need memory allocation due to the array size.
	// We will have 2 images for training (neutral expression) and 12 to test.
	Mat trainingDataMat(52 * 2, 512, CV_32F);
	Mat labelsMat(52 * 2, 1, CV_32S);
	Mat testDataMat(52 * 12, 512, CV_32F);
	Mat labelsTest(52 * 12, 1, CV_32S);
	string current_filename;
	
	cout << "Prepare for feature extraction ... " << endl << endl; 

	int training_samples_visited = 0;
	int test_samples_visited = 0;

	for (int i = 0; i < files.size(); i++){
		current_filename = files[i].substr(files[i].size()-6,2);
		if (current_filename != "al"){ 
			// If image isn't neutral save in testDataMat
			image = imread(files[i], 0);
			
			labelsTest.at<int>(test_samples_visited) = (int) test_samples_visited / 12 + 1;
			// Get a 512 feature vector
			hog.compute(image, descriptors, Size(1, 1), Size(0, 0));

			// Save feature vector in the trainingData
			for (int j = 0; j < descriptors.size(); j++){
				testDataMat.at<float>(test_samples_visited, j) = descriptors[j];
			}

			test_samples_visited++;
		}
		else{
			// If image is neutral save in trainingDataMat
			image = imread(files[i], 0);

			labelsMat.at<int>(training_samples_visited) = (int) training_samples_visited / 2 + 1;
			// Get a 512 feature vector
			hog.compute(image, descriptors, Size(1, 1), Size(0, 0));

			// Save feature vector in the trainingData
			for (int j = 0; j < descriptors.size(); j++){
				trainingDataMat.at<float>(training_samples_visited, j) = descriptors[j];
			}

			training_samples_visited++;
		}


		cout << "Feature Extraction of image " << i + 1 << " of a total of " << 52 * 14 << endl;
	}

	cout << endl <<  "Feature Extraction ended successfully!" << endl << endl;

	// Prepare the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	float best_accuracy = 0;
	float best_Gamma = 0;
	float best_C = 0;
	float current_C, current_gamma;
	float current_accuracy;

	cout << "Starting dense-grid parameter estimation for C-SVM, using a RBF kernel... " << endl << endl;
	
	int current_model = 1;

	Mat predictedLabels(52 * 12, 1, CV_32S);

	// Dense grid between 10^-15 and 10^15 for gamma and C
	for (int i = -15; i < 16; i++){
		for (int j = -15; j < 16; j++){
			current_C = (float) pow(10, i);
			current_gamma = (float) pow(10, j);
			svm->setC(current_C);
			svm->setGamma(current_gamma);
			

			// Train Model
			svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
			svm->predict(testDataMat, predictedLabels);

			current_accuracy = getClassifierAccuracy(predictedLabels, labelsTest) * 100;

			cout << "Training model " << current_model << " of " << 31 * 31 << ". (C = " << current_C << " , G = " <<
				current_gamma << ") - Accuraccy: " << current_accuracy << " %."<< endl;

			if (current_accuracy >= best_accuracy){
				best_C = current_C;
				best_Gamma = current_gamma;
				best_accuracy = current_accuracy;
			}

			current_model++;
		}
	}

	cout << "Dense Grid estimation ended! " << endl << endl;

	cout << "Best C: " << best_C << endl;
	cout << "Best Gamma: " << best_Gamma << endl;
	cout << "Best Accuracy: " << best_accuracy << " %" << endl << endl;
	
	// Train and save best model.
	svm->setC(best_C);
	svm->setGamma(best_Gamma);
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->save("Best_Model.xml");

	system("Pause");
	return 0;
}
