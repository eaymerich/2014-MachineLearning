//============================================================================
// Name        : mp3.cpp
// Author      : Edward Aymerich
// Version     :
// Copyright   : University of Central Florida, 2014
// Description : Hello World in C++, Ansi-style
//============================================================================

#define EIGEN_MPL2_ONLY 1

#include <iostream>
#include <fstream>
#include <string>
#include "Eigen\Dense"
#include "MyClock.h"
#include "MyLog.h"
#include "PCA.h"
#include "FDA.h"
#include "MatrixUtil.h"
#include "SampleD.h"
#include "KnnClassifier.h"
#include "NgbClassifier.h"

using namespace Eigen;
using namespace std;

MyLog mlog("log.txt");

void read_uint32(ifstream &in, uint32_t &n){
	in.read((char*)&n, sizeof(n));
	n = __builtin_bswap32(n);
}

void loadLabels(string label_filename, vector<uint8_t>& label){

	// IDX data
	uint32_t label_magic_number;
	uint32_t labels;

	ifstream in_label(label_filename, ios_base::in | ios_base::binary);
	read_uint32(in_label,label_magic_number);
	if(label_magic_number != 2049){
		cerr << "ERROR: label file doesn't start with magic number." << endl;
		return;
	}
	read_uint32(in_label,labels);

	//cout << "Label Magic Number= " << label_magic_number << endl;
	//cout << "# of labels= " << labels << endl;

	// Read labels
	label = move(vector<uint8_t>(labels));
	in_label.read((char*)(label.data()), sizeof(uint8_t)*labels);

	in_label.close();
}

void loadData(string sample_filename, MatrixXd &m){

	// IDX data
	uint32_t sample_magic_number;
	uint32_t samples;
	uint32_t sample_height;
	uint32_t sample_width;

	ifstream in_sample(sample_filename, ios_base::in | ios_base::binary);
	read_uint32(in_sample,sample_magic_number);
	if(sample_magic_number != 2051){
		mlog << endl << "ERROR: sample file doesn't start with magic number." << endl;
		return;
	}
	read_uint32(in_sample,samples);
	read_uint32(in_sample,sample_height);
	read_uint32(in_sample,sample_width);

	//cout << "Sample Magic Number= " << sample_magic_number << endl;
	//cout << "# of samples= " << samples << endl;
	//cout << "Sample height= " << sample_height << endl;
	//cout << "Sample width= " << sample_width << endl;

	// Create matrix.
	m = move( MatrixXd(sample_width*sample_height, samples) );

	// Read samples
	for(uint32_t s=0; s < samples; s++){

		// Read sample vector
		vector<uint8_t>buff(sample_width*sample_height);
		if(!in_sample.read((char*)&(buff[0]),sample_height*sample_width)){
			//cerr << "Fail reading data from file: " << sample_filename << endl;
			mlog << endl << "ERROR: Fail reading data from file: " << sample_filename << endl;
		}

		for(unsigned int i = 0; i < sample_width*sample_height; i++){
			m(i,s) = ((double)buff[i]) / 255.0;
		}

	}

	in_sample.close();
}

void knnParallelTest(vector<SampleD> &trainset, vector<SampleD> &testset, unsigned int k, unsigned int *error){
	KnnClassifier knn(trainset,k);
	unsigned int i,p,errors=0;

	#pragma omp parallel for schedule(static) private(i,p) reduction(+:errors)
	for(i = 0; i < testset.size(); i++){
		p = knn.classify(testset[i]);
		if(p != testset[i].label){
			errors++;
		}
	}

	*error = errors;
}

void knnTestProtocol1(vector<SampleD> &trainset, vector<SampleD> &testset){
	mlog << "--------------------------------------------------------------------------------" << endl;
	mlog << "*** KNN Processing all tests for Test Protocol 1 ***" << endl;
	mlog << "Protocol 1 - training set: " << trainset.size() << " - test set: " << testset.size() << endl;
	MyClock clock;
	mlog << "Testing k values:" << endl;
	vector<unsigned int> k_values{1,2,3,4,5,6,7,8,9,10,20,30,50,100};
	mlog << "k,error(%),time(ms)" << endl;
	for(auto k : k_values){
		mlog << k << ","; flush(mlog);
		unsigned int error = 0;
		clock.start();
		//knnTestProtocol1( trainset, testset, k, &error);
		knnParallelTest( trainset, testset, k, &error);
		clock.stop();
		mlog << (double(error) / (double)testset.size())*100.0 << "%,";
		mlog << clock.getMilliseconds() << endl;
	}
}

void knnTestProtocol2(vector<SampleD> &trainset, vector<SampleD> &testset, unsigned int val){
	mlog << "--------------------------------------------------------------------------------" << endl;
	mlog << "*** KNN Processing all tests for Test Protocol 2 ***" << endl;
	MyClock clock,clock2;

	// Split trainset into trainset and validation set
	mlog << "Splitting trainset into trainset/validationset... ";
	clock.start();
	unsigned int newtraintset_size = trainset.size() - val;
	vector<SampleD> newtrainset(trainset.begin(), trainset.begin() + newtraintset_size);
	vector<SampleD> valset(trainset.begin() + newtraintset_size, trainset.end());
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

	mlog << "Protocol 2 - training set: " << newtrainset.size() << " - validation set: " << valset.size() << " - test set: " << testset.size() << endl;

	// Tune parameter k on validation set
	mlog << "Tuning parameter k:" << endl;
	vector<unsigned int> k_values{1,2,3,4,5,6,7,8,9,10};
	unsigned int best_k=1;
	unsigned int best_error = numeric_limits<unsigned int>::max();
	unsigned int error;
	clock.start();
	mlog << "k,error(%),time(ms)" << endl;
	for(auto k : k_values){
		mlog << k << ","; flush(mlog);
		clock2.start();
		knnParallelTest(newtrainset, valset, k, &error);
		clock2.stop();
		mlog << (double(error) / (double)testset.size())*100.0 << "%,";
		mlog << ((double)clock2.getMilliseconds()) << endl;
		if(error < best_error){
			best_k = k;
			best_error = error;
		}
	}
	clock.stop();
	mlog << "Best value for k on validation set: " << best_k << ". Found in " << (double)clock.getMilliseconds() << "ms." << endl;

	// Use best k on test set
	mlog << "Testing best k on test set..." << endl;
	mlog << best_k << ","; flush(mlog);
	clock.start();
	knnParallelTest(newtrainset, testset, best_k, &error);
	clock.stop();
	mlog << (double(error) / (double)testset.size())*100.0 << "%,";
	mlog << ((double)clock2.getMilliseconds()) << endl;

	newtrainset.clear();
	valset.clear();

	//log << "All " << testset.size() << " test samples processed in " << ((double)clock.getMilliseconds())/1000.0 << "s." << endl;
	//log << "Error rate: " << (double(error) / (double)testset.size())*100.0 << "%" << endl;
}

void knnTestProtocol3(vector<SampleD> &trainset, vector<SampleD> &testset, unsigned int fold){
	mlog << "--------------------------------------------------------------------------------" << endl;
	mlog << "*** KNN Processing all tests for Test Protocol 3 ***" << endl;
	MyClock clock,clock2;

	vector<SampleD> valset;
	vector<SampleD> trainsubset;
	unsigned int error;
	unsigned int errors = 0;
	unsigned int block_size = trainset.size() / fold;
	vector<unsigned int> k_values{1,3,5,7,9};

	mlog << "Protocol 3 - training set: " << trainsubset.size() << " - validation set: " << block_size << " - " << fold << "-fold" << endl;

	for(auto k : k_values){
		mlog << "Testing k=" << k << "..." << endl;
		errors = 0;
		clock.start();
		mlog << "fold,error(%),time(ms)" << endl;
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
			mlog << i+1 << ","; flush(mlog);
			clock2.start();
			knnParallelTest(trainsubset, valset, k, &error);
			clock2.stop();
			mlog << (double(error) / (double)block_size)*100.0 << "%,";
			mlog << clock2.getMilliseconds() << endl;
			errors += error;

			// Reset sets
			trainsubset.clear();
			valset.clear();
		}
		clock.stop();
		double avg_error = ((double)errors / (double)(block_size * fold)) * 100.0;
		mlog << "k=" << k << " avg_error=" << avg_error << "%";
		mlog << " time=" << clock.getMilliseconds() << "ms." << endl;
	}
}

void ngbTest(vector<SampleD> &trainset, vector<SampleD> &testset, double alpha, unsigned int *error){

	NgbClassifier ngb(trainset, alpha);
	ngb.train();

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

void ngbTestProtocol1(vector<SampleD> &trainset, vector<SampleD> &testset){
	mlog << "--------------------------------------------------------------------------------" << endl;
	mlog << "*** NGB Processing all tests for Test Protocol 1 ***" << endl;
	mlog << "Protocol 1 - training set: " << trainset.size() << " - test set: " << testset.size() << endl;
	MyClock clock;
	mlog << "Testing alpha values:" << endl;
	//vector<double> alpha_values{0,1,2,4,8,16};
	vector<double> alpha_values{0.0};
	mlog << "alpha,error,time(ms)" << endl;
	//unsigned int error = 0;
	//ngbTest( trainset, testset, 0.0, &error);

	for(auto alpha : alpha_values){
		mlog << alpha << ","; flush(mlog);
		unsigned int error = 0;
		clock.start();
		ngbTest( trainset, testset, alpha, &error);
		clock.stop();
		mlog << (double(error) / (double)testset.size())*100.0 << "%,";
		mlog << clock.getMilliseconds() << endl;
	}
}


void ngbTestProtocol3(vector<SampleD> &trainset, vector<SampleD> &testset, unsigned int fold){
	mlog << "--------------------------------------------------------------------------------" << endl;
	mlog << "*** NGB Processing all tests for Test Protocol 3 ***" << endl;
	MyClock clock,clock2;

	vector<SampleD> valset;
	vector<SampleD> trainsubset;
	unsigned int error;
	unsigned int errors = 0;
	unsigned int block_size = trainset.size() / fold;
	//vector<double> alpha_values{1,2,4,8,16};
	vector<double> alpha_values{0.0};

	mlog << "Protocol 3 - training set: " << trainsubset.size() << " - validation set: " << block_size << " - " << fold << "-fold" << endl;

	for(auto alpha : alpha_values){
		mlog << "Testing alpha=" << alpha << "..." << endl;
		errors = 0;
		clock.start();
		mlog << "fold,error,time(ms)" << endl;
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
			mlog << i+1 << ","; flush(mlog);
			clock2.start();
			ngbTest(trainsubset, valset, alpha, &error);
			clock2.stop();
			mlog << (double(error) / (double)block_size)*100.0 << "%,";
			mlog << clock2.getMilliseconds() << endl;
			errors += error;

			// Reset sets
			trainsubset.clear();
			valset.clear();
		}
		clock.stop();
		double avg_error = ((double)errors / (double)(block_size * fold)) * 100.0;
		mlog << "alpha=" << alpha << " avg_error=" << avg_error << "%";
		mlog << " time=" << clock.getMilliseconds() << "ms." << endl;
	}
}

void transformData(){
	MyClock clock;
	mlog << "<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>" << endl;
	mlog << "<< Reducing MNIST dimensionality" << endl;

	Eigen::MatrixXd x_train, x_test, y;
	vector<uint8_t> label;

	mlog << "Loading data..."; flush(mlog);
	clock.start();
	loadData("../MNIST/t10k-images.idx3-ubyte", x_test);
	loadData("../MNIST/train-images.idx3-ubyte", x_train);
	loadLabels("../MNIST/train-labels.idx1-ubyte", label);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	mlog << "Matrix X (train) has a size of " << x_train.rows() << "x" << x_train.cols() << endl;
	mlog << "Matrix X (test) has a size of " << x_test.rows() << "x" << x_test.cols() << endl;
	mlog << "labels loaded: " << label.size() << endl;

	mlog << "<< Aplying PCA" << endl;
	PCA pca{mlog};
	pca.train(x_train);
	mlog << "Computing Y (train set)..."; flush(mlog);
	clock.start();
	pca.transform(x_train,y);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	writeMatrix("../MNIST/pca/pca_train.dat", y);
	mlog << "Computing Y (test set)..."; flush(mlog);
	clock.start();
	pca.transform(x_test,y);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	writeMatrix("../MNIST/pca/pca_test.dat", y);

	mlog << "<< Aplying FDA" << endl;
	FDA fda{mlog};
	fda.train(x_train,label);
	mlog << "Computing Y (train set)..."; flush(mlog);
	clock.start();
	fda.transform(x_train,y);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	writeMatrix("../MNIST/fda/fda_train.dat", y);
	mlog << "Computing Y (test set)..."; flush(mlog);
	clock.start();
	fda.transform(x_test,y);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	writeMatrix("../MNIST/fda/fda_test.dat", y);

	mlog << "<< Finish reducing MNIST dimensionality." << endl << endl;
}

void testData(){
	MyClock clock;
	mlog << "<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>" << endl;
	mlog << "<< Testing MNIST PCA-reduced dataset" << endl;
	vector<SampleD> trainset, testset;
	mlog << "Loading data..."; flush(mlog);
	clock.start();
	SampleD::loadSamplesFromMatrix("../MNIST/pca/pca_train.dat", "../MNIST/train-labels.idx1-ubyte", trainset);
	SampleD::loadSamplesFromMatrix("../MNIST/pca/pca_test.dat", "../MNIST/t10k-labels.idx1-ubyte", testset);
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

	knnTestProtocol1(trainset, testset);
	knnTestProtocol2(trainset, testset, 10000);
	knnTestProtocol3(trainset, testset, 5);
	knnTestProtocol3(trainset, testset, 10);

	ngbTestProtocol1(trainset, testset);
	ngbTestProtocol3(trainset, testset, 5);
	ngbTestProtocol3(trainset, testset, 10);

	mlog << endl;
	mlog << "<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>" << endl;
	mlog << "<< Testing MNIST FDA-reduced dataset" << endl;
	SampleD::loadSamplesFromMatrix("../MNIST/fda/fda_train.dat", "../MNIST/train-labels.idx1-ubyte", trainset);
	SampleD::loadSamplesFromMatrix("../MNIST/fda/fda_test.dat", "../MNIST/t10k-labels.idx1-ubyte", testset);
	knnTestProtocol1(trainset, testset);
	knnTestProtocol2(trainset, testset, 10000);
	knnTestProtocol3(trainset, testset, 5);
	knnTestProtocol3(trainset, testset, 10);

	ngbTestProtocol1(trainset, testset);
	ngbTestProtocol3(trainset, testset, 5);
	ngbTestProtocol3(trainset, testset, 10);
}

int main() {
	Eigen::initParallel();
	mlog << "****************************************" << endl;
	transformData();
	testData();
	cout << "Finish Machine Problem 3!" << endl;
	return 0;
}
