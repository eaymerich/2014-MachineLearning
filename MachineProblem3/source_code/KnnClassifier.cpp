/*
 * KnnClassifier.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#include <vector>
using namespace std;

#include "Champion.h"
#include "KnnClassifier.h"


KnnClassifier::KnnClassifier(vector<SampleD> &t, unsigned int knn)
	: trainset(t), k(knn){
}

KnnClassifier::~KnnClassifier() {

}

unsigned int KnnClassifier::classify(SampleD &s){
	// Look for closest neighbors
	Champion closest(k);
	double dist;
	for(unsigned int i=0; i < trainset.size(); i++){
		dist = s.distance(trainset[i]);
		closest.testContestant(i, dist);
	}

	// Get consensus on neighbors
	vector<unsigned int> vote(10,0);
	for(unsigned int i=0; i < closest.size(); i++){
		// Count labels
		vote[(trainset[closest.getChampion(i)]).label]++;
	}
	unsigned int p=0;
	unsigned int max_count = 0;
	for(unsigned int i=0; i < vote.size(); i++){
		// Find winner
		if(vote[i] > max_count){
			max_count = vote[i];
			p = i;
		}
	}
	return p;
}

