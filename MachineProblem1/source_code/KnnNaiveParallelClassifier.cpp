/*
 * KnnNaiveParallelClassifier.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#include <iostream>
#include <vector>
#include <thread>
using namespace std;

#include "Champion.h"
#include "KnnNaiveParallelClassifier.h"

struct finder{
	vector<Sample> &trainset;
	Sample &s;
	unsigned int start;
	unsigned int end;
	Champion &closest;
	void operator()(){
		double dist;
		for(unsigned int i=start; i < end && i < trainset.size(); i++){
			dist = s.distance(trainset[i]);
			closest.testContestant(i, dist);
		}
	}
};

KnnNaiveParallelClassifier::KnnNaiveParallelClassifier(vector<Sample> &t, unsigned int knn)
	: KnnClassifier(t,knn) {

}

KnnNaiveParallelClassifier::~KnnNaiveParallelClassifier() {

}

unsigned int KnnNaiveParallelClassifier::classify(Sample &s){
	//unsigned int procs = 1;
	unsigned int procs = thread::hardware_concurrency();
	//cout << "Concurrency=" << procs << endl;

	vector<Champion> champs(procs,Champion(k));

	// Spawn procs threads to look for the closest neighbors
	vector<thread> t(procs);
	unsigned int block_size = trainset.size() / procs;
	for(unsigned int i = 0; i < t.size(); i++){
		/*
		finder f{trainset,s,i*block_size, (i*block_size)+block_size, champs[i]};
		//cout << "finder look from:" << f.start << " until:" << f.end << endl;
		thread tmp{f};
		t[i] = move(tmp);
		*/
		t[i] = thread{finder{trainset,s,i*block_size, (i*block_size)+block_size, champs[i]}};
	}
	for(unsigned int i = 0; i < t.size(); i++){
		t[i].join();
	}

	// Unify champions
	Champion closest(k);
	for(unsigned int i=0; i < champs.size(); i++){
		closest.combine(champs[i]);
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

