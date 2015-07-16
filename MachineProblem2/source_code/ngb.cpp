//============================================================================
// Name        : ngb.cpp
// Author      : Edward Aymerich
// Version     :
// Copyright   : UCF 2014
// Description : Naive Gaussian Bayes classifier.
//               Machine Problem 2
//               COP-5610 Machine Learning
//               Fall 2014
//============================================================================

#include <iostream>
using namespace std;

#include "Mylog.h"
#include "MyClock.h"
#include "NgbClassifier.h"

MyLog mylog("mylog.txt");

void ngbTest(vector<Sample> &trainset, vector<Sample> &testset, double alpha, unsigned int *error){

	//MyClock clock;
	NgbClassifier ngb(trainset, alpha);
	//mylog << "Training classifier...";
	//mylog.flush();
	//clock.start();
	ngb.train();
	//clock.stop();
	//mylog << "Ok. Training time: " << clock.getMilliseconds() << "ms." << endl;

	unsigned int errors = 0;
	unsigned int i;
	unsigned int p;

	#pragma omp parallel for schedule(static) private(i,p) reduction(+:errors)
	for(i = 0; i < testset.size(); i++){
		p = ngb.classify(testset[i]);
		if(p != testset[i].label){
			errors++;
		}
	}

	*error = errors;
}

void ngbTestProtocol1(vector<Sample> &trainset, vector<Sample> &testset){
	mylog << "--------------------------------------------------------------------------------" << endl;
	mylog << "*** Processing all tests for Test Protocol 1 ***" << endl;
	mylog << "Protocol 1 - training set: " << trainset.size() << " - test set: " << testset.size() << endl;
	MyClock clock;
	mylog << "Testing alpha values:" << endl;
	vector<double> alpha_values{0,1,2,4,8,16};
	//vector<double> alpha_values{0.0};
	for(auto alpha : alpha_values){
		mylog << "alpha=" << alpha;
		mylog.flush();
		unsigned int error = 0;
		clock.start();
		ngbTest( trainset, testset, alpha, &error);
		clock.stop();
		mylog << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
		mylog << " time=" << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	}
}


void ngbTestProtocol2(vector<Sample> &trainset, vector<Sample> &testset, unsigned int val){
	mylog << "--------------------------------------------------------------------------------" << endl;
	mylog << "*** Processing all tests for Test Protocol 2 ***" << endl;
	MyClock clock,clock2;

	// Split trainset into trainset and validation set
	mylog << "Splitting trainset into trainset/validationset... ";
	clock.start();
	unsigned int newtraintset_size = trainset.size() - val;
	vector<Sample> newtrainset(trainset.begin(), trainset.begin() + newtraintset_size);
	vector<Sample> valset(trainset.begin() + newtraintset_size, trainset.end());
	clock.stop();
	mylog << "Done in " << clock.getMilliseconds() << "ms." << endl;

	mylog << "Protocol 2 - training set: " << newtrainset.size() << " - validation set: " << valset.size() << " - test set: " << testset.size() << endl;

	// Tune parameter k on validation set
	mylog << "Tuning parameter k:" << endl;
	vector<double> alpha_values{1,2,4,8,16};
	double best_alpha=1;
	unsigned int best_error = numeric_limits<unsigned int>::max();
	unsigned int error;
	clock.start();
	for(auto alpha : alpha_values){
		mylog << "alpha=" << alpha;
		mylog.flush();
		clock2.start();
		ngbTest(newtrainset, valset, alpha, &error);
		clock2.stop();
		mylog << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
		mylog << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;
		if(error < best_error){
			best_alpha = alpha;
			best_error = error;
		}
	}
	clock.stop();
	mylog << "Best value for k on validation set: " << best_alpha << ". Found in " << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;

	// Use best k on test set
	mylog << "Testing best k on test set..." << endl;
	mylog << "alpha=" << best_alpha;
	clock.start();
	ngbTest(newtrainset, testset, best_alpha, &error);
	clock.stop();
	mylog << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
	mylog << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;

	newtrainset.clear();
	valset.clear();

	//log << "All " << testset.size() << " test samples processed in " << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	//log << "Error rate: " << (double(error) / (double)testset.size())*100.0 << "%" << endl;
}


void ngbTestProtocol3(vector<Sample> &trainset, vector<Sample> &testset, unsigned int fold){
	mylog << "--------------------------------------------------------------------------------" << endl;
	mylog << "*** Processing all tests for Test Protocol 3 ***" << endl;
	MyClock clock,clock2;

	vector<Sample> valset;
	vector<Sample> trainsubset;
	unsigned int error;
	unsigned int errors = 0;
	unsigned int block_size = trainset.size() / fold;
	vector<double> alpha_values{1,2,4,8,16};

	mylog << "Protocol 3 - training set: " << trainsubset.size() << " - validation set: " << block_size << " - " << fold << "-fold" << endl;

	for(auto alpha : alpha_values){
		mylog << "Testing alpha=" << alpha << "..." << endl;
		errors = 0;
		clock.start();
		for(unsigned int i = 0; i < fold; i++){
			// Create test subset and validation set
			//log << "Splitting trainset into trainset/validationset... ";
			//clock2.start();
			valset.insert(valset.end(), trainset.begin() + (i*block_size), trainset.begin() + (i*block_size) + block_size);
			trainsubset.insert(trainsubset.end(), trainset.begin(), trainset.begin() + (i*block_size)); // Copy samples before val
			trainsubset.insert(trainsubset.end(), trainset.begin() + (i*block_size) + block_size, trainset.end()); // Copy samples after val
			//clock2.stop();
			//log << "Done in " << clock2.getMilliseconds() << "ms." << endl;

			// Test
			mylog << "fold=" << i+1;
			mylog.flush();
			clock2.start();
			ngbTest(trainsubset, valset, alpha, &error);
			clock2.stop();
			mylog << " error=" << (double(error) / (double)block_size)*100.0 << "%";
			mylog << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;
			errors += error;

			// Reset sets
			trainsubset.clear();
			valset.clear();
		}
		clock.stop();
		double avg_error = ((double)errors / (double)(block_size * fold)) * 100.0;
		mylog << "alpha=" << alpha << " avg_error=" << avg_error << "%";
		mylog << " time=" << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	}


}

int main() {
	mylog << "****************************************" << endl;
	mylog << "Naive Gaussian Bayes classifier." << endl;

	vector<Sample> trainset;
	vector<Sample> testset;
	MyClock clock;

	// Load training set from file.
	mylog << "Loading training samples...";
	mylog.flush();
	clock.start();
	Sample::loadSamples("train-images.idx3-ubyte","train-labels.idx1-ubyte", trainset);
	clock.stop();
	mylog << "OK. Load time: " << clock.getMilliseconds() <<	"ms." << endl;

	// Load test set from file.
	mylog << "Loading test samples...";
	mylog.flush();
	clock.start();
	Sample::loadSamples("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte", testset);
	clock.stop();
	mylog << "OK. Load time: " << clock.getMilliseconds() <<	"ms." << endl;


	ngbTestProtocol1(trainset, testset);
	ngbTestProtocol2(trainset, testset,10000);
	ngbTestProtocol3(trainset, testset, 5);
	ngbTestProtocol3(trainset, testset, 10);

	/*
	// Train Classifier
	mylog << "Training NGB classifier...";
	mylog.flush();
	NgbClassifier ngb(trainset);
	clock.start();
	ngb.train();
	clock.stop();
	mylog << "OK. Training time: " << clock.getMilliseconds() <<	"ms." << endl;

	// Test samples in testset.
	unsigned int errors = 0;
	for(unsigned int i = 0; i < testset.size(); i++){
		//clock.start();
		unsigned int label = ngb.classify(testset[i]);
		if (label != testset[i].label){ ++errors;}
		//clock.stop();
		//mylog << "Class=" << label << endl;
		//mylog << "OK. Classification time: " << clock.getMilliseconds() <<	"ms." << endl;
		//mylog << "Real label=" << (unsigned int)testset[i].label << endl;
	}
	mylog << " errors=" << (double)errors / (double)testset.size() << endl;
	 */

	return 0;
}
