/*
 * NgbClassifier.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Edward
 */

#include <iostream>
#include <limits>
#include <cmath>

#include "NgbClassifier.h"

using namespace std;

NgbClassifier::NgbClassifier(std::vector<Sample> &t, double a) :
	trainset(t), alpha(a), prior(10) {

}

NgbClassifier::~NgbClassifier() {
	prior.clear();
	mean.clear();
	variance.clear();
}

unsigned int NgbClassifier::classify(Sample &s){

	double max_posterior = -1.0 * numeric_limits<double>::max();
	unsigned int label = 0;

	//Calculate posterior for all classes and choose the largest.
	for(unsigned int i = 0; i < 10; i++){
		double likelihood = SampleD::likelihood(s,mean[i],variance[i]);
		double posterior = likelihood * prior[i];
		//double posterior = likelihood + log(prior[i]);

		//cout << "class=" << i << " like=" << likelihood << " prior=" << prior[i] << " posterior=" << posterior << endl;

		if(posterior > max_posterior){
			max_posterior = posterior;
			label = i;
		}
	}

	return label;
}

void NgbClassifier::train(){

	// Count how many samples are in each class
	vector<unsigned int> n(10,0);
	for(unsigned int i = 0; i < trainset.size(); i++){
		++n[trainset[i].label];
	}

	// Estimate prior
	if(alpha == 0.0){
		// MLE
		for(unsigned int i = 0; i < prior.size(); i++){
			prior[i] = n[i] / static_cast<double>( trainset.size() );
		}
	}else{
		// MAP
		double alpha_d = static_cast<double>(trainset.size()) * alpha / 100.0;
		double alpha_sum = alpha_d * 10.0;
		for(unsigned int i = 0; i < prior.size(); i++){
			prior[i] = (n[i] + alpha_d) / (static_cast<double>(trainset.size()) + alpha_sum);
		}
	}

	// Calculate mean and variance
	SampleD zeros(28*28);
	zeros = 0.0;
	vector<SampleD> sums(10,zeros);
	vector<SampleD> squares(10,zeros);
	for(unsigned int i = 0; i < trainset.size(); i++){
		unsigned int label = trainset[i].label;
		SampleD::sum_and_square(trainset[i],sums[label], squares[label]);
	}

	for(unsigned int i = 0; i < sums.size(); i++){
		mean.push_back(zeros);
		variance.push_back(zeros);
		SampleD::mean_and_variance(n[i],sums[i],squares[i],mean[i],variance[i]);
	}

}

void NgbClassifier::incrementVariance(){
	for(unsigned int i = 0; i < 10; i++){
		variance[i] += 0.01;
	}
}
