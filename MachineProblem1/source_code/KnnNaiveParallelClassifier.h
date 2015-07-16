/*
 * KnnNaiveParallelClassifier.h
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#ifndef KNNNAIVEPARALLELCLASSIFIER_H_
#define KNNNAIVEPARALLELCLASSIFIER_H_

#include "KnnClassifier.h"
#include "Champion.h"

class KnnNaiveParallelClassifier: public KnnClassifier {
public:
	KnnNaiveParallelClassifier(std::vector<Sample> &t, unsigned int knn=1);
	virtual ~KnnNaiveParallelClassifier();
	virtual unsigned int classify(Sample &s);
};

#endif /* KNNNAIVEPARALLELCLASSIFIER_H_ */
