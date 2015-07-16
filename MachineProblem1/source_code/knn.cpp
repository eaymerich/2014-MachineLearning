//============================================================================
// Name        : knn.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <thread>
#include <stdint.h>
using namespace std;

#include "MyLog.h"
#include "Sample.h"
#include "Champion.h"
#include "MyClock.h"
#include "KnnClassifier.h"

MyLog log("log.txt");

struct Tester{
	vector<Sample> &trainset;
	vector<Sample> &testset;
	unsigned int start;
	unsigned int end;
	unsigned int k;
	unsigned int *errors;
	void operator()(){
		KnnClassifier knn(trainset,k);
		*errors = 0;
		for(unsigned int j = start; j < end && j < testset.size(); j++){
			unsigned int p = knn.classify(testset[j]);
			if(p != testset[j].label){
				(*errors)++;
			}
		}
	}
};

void knnParallelTest(vector<Sample> &trainset, vector<Sample> &testset, unsigned int k, unsigned int *error);
void knnTest(vector<Sample> &trainset, vector<Sample> &testset, unsigned int k, unsigned int *error);
void knnTestProtocol1(vector<Sample> &trainset, vector<Sample> &testset);
void knnTestProtocol2(vector<Sample> &trainset, vector<Sample> &testset, unsigned int val);

void knnTest(vector<Sample> &trainset, vector<Sample> &testset, unsigned int k, unsigned int *error){
	KnnClassifier knn(trainset,k);
	for(unsigned int j = 0; j < testset.size(); j++){
		unsigned int p = knn.classify(testset[j]);
		if(p != testset[j].label){
			(*error)++;
		}
	}
}

void knnParallelTest(vector<Sample> &trainset, vector<Sample> &testset, unsigned int k, unsigned int *error){
	//vector<unsigned int> k_values{1,3,5,10,20,30,50,100};

	unsigned int procs = thread::hardware_concurrency();

	// Spawn procs threads to look for the closest neighbors
	vector<thread> t;
	vector<Tester> testers;
	vector<unsigned int> errors(procs);
	unsigned int block_size = testset.size() / procs;
	//unsigned int block_size = 100;
	for(unsigned int i = 0; i < procs; i++){
		testers.push_back(Tester{trainset,testset,i*block_size, (i*block_size)+block_size,k,&(errors[i])});
	}
	for(unsigned int i = 0; i < procs; i++){
		t.push_back( thread{testers[i]} );
	}
	for(auto& tt : t){
		tt.join();
	}

	// Unify errors
	*error = 0;
	for(auto e : errors){
		*error += e;
	}

}

void knnTestProtocol1(vector<Sample> &trainset, vector<Sample> &testset){
	log << "--------------------------------------------------------------------------------" << endl;
	log << "*** Processing all tests for Test Protocol 1 ***" << endl;
	log << "Protocol 1 - training set: " << trainset.size() << " - test set: " << testset.size() << endl;
	MyClock clock;
	log << "Testing k values:" << endl;
	vector<unsigned int> k_values{1,2,3,4,5,6,7,8,9,10,20,30,50,100};
	for(auto k : k_values){
		log << "k=" << k;
		log.flush();
		unsigned int error = 0;
		clock.start();
		//knnTestProtocol1( trainset, testset, k, &error);
		knnParallelTest( trainset, testset, k, &error);
		clock.stop();
		log << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
		log << " time=" << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	}
}

void knnTestProtocol2(vector<Sample> &trainset, vector<Sample> &testset, unsigned int val){
	log << "--------------------------------------------------------------------------------" << endl;
	log << "*** Processing all tests for Test Protocol 2 ***" << endl;
	MyClock clock,clock2;

	// Split trainset into trainset and validation set
	log << "Splitting trainset into trainset/validationset... ";
	clock.start();
	unsigned int newtraintset_size = trainset.size() - val;
	vector<Sample> newtrainset(trainset.begin(), trainset.begin() + newtraintset_size);
	vector<Sample> valset(trainset.begin() + newtraintset_size, trainset.end());
	clock.stop();
	log << "Done in " << clock.getMilliseconds() << "ms." << endl;

	log << "Protocol 2 - training set: " << newtrainset.size() << " - validation set: " << valset.size() << " - test set: " << testset.size() << endl;

	// Tune parameter k on validation set
	log << "Tuning parameter k:" << endl;
	vector<unsigned int> k_values{1,2,3,4,5,6,7,8,9,10};
	unsigned int best_k=1;
	unsigned int best_error = numeric_limits<unsigned int>::max();
	unsigned int error;
	clock.start();
	for(auto k : k_values){
		log << "k=" << k;
		log.flush();
		clock2.start();
		knnParallelTest(newtrainset, valset, k, &error);
		clock2.stop();
		log << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
		log << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;
		if(error < best_error){
			best_k = k;
			best_error = error;
		}
	}
	clock.stop();
	log << "Best value for k on validation set: " << best_k << ". Found in " << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;

	// Use best k on test set
	log << "Testing best k on test set..." << endl;
	log << "k=" << best_k;
	clock.start();
	knnParallelTest(newtrainset, testset, best_k, &error);
	clock.stop();
	log << " error=" << (double(error) / (double)testset.size())*100.0 << "%";
	log << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;

	newtrainset.clear();
	valset.clear();

	//log << "All " << testset.size() << " test samples processed in " << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	//log << "Error rate: " << (double(error) / (double)testset.size())*100.0 << "%" << endl;
}

void knnTestProtocol3(vector<Sample> &trainset, vector<Sample> &testset, unsigned int fold){
	log << "--------------------------------------------------------------------------------" << endl;
	log << "*** Processing all tests for Test Protocol 3 ***" << endl;
	MyClock clock,clock2;

	vector<Sample> valset;
	vector<Sample> trainsubset;
	unsigned int error;
	unsigned int errors = 0;
	unsigned int block_size = trainset.size() / fold;
	vector<unsigned int> k_values{1,3,5,7,9};

	log << "Protocol 3 - training set: " << trainsubset.size() << " - validation set: " << block_size << " - " << fold << "-fold" << endl;

	for(auto k : k_values){
		log << "Testing k=" << k << "..." << endl;
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
			log << "fold=" << i+1;
			log.flush();
			clock2.start();
			knnParallelTest(trainsubset, valset, k, &error);
			clock2.stop();
			log << " error=" << (double(error) / (double)block_size)*100.0 << "%";
			log << " time=" << ((double)clock2.getMilliseconds())/1000.0 << "s." << endl;
			errors += error;

			// Reset sets
			trainsubset.clear();
			valset.clear();
		}
		clock.stop();
		double avg_error = ((double)errors / (double)(block_size * fold)) * 100.0;
		log << "k=" << k << " avg_error=" << avg_error << "%";
		log << " time=" << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	}


}


int main() {
	log << "Hello KNN!!!" << endl; // prints !!!Hello World!!!

	vector<Sample> trainset;
	vector<Sample> testset;
	MyClock clock;

	log << "Loading training samples...";
	log.flush();
	clock.start();
	Sample::loadSamples("train-images.idx3-ubyte","train-labels.idx1-ubyte", trainset);
	clock.stop();
	log << "OK. Load time: " << clock.getMilliseconds() <<	"ms." << endl;


	log << "Loading test samples...";
	log.flush();
	clock.start();
	Sample::loadSamples("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte", testset);
	clock.stop();
	log << "OK. Load time: " << clock.getMilliseconds() <<	"ms." << endl;

	knnTestProtocol1(trainset, testset);
	knnTestProtocol2(trainset, testset, 10000);
	knnTestProtocol3(trainset, testset, 5);
	knnTestProtocol3(trainset, testset, 10);
	return 0;
}
