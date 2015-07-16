/*
 * KnnClassifier.h
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#ifndef KNNCLASSIFIER_H_
#define KNNCLASSIFIER_H_

#include <vector>
#include "Sample.h"

class KnnClassifier {
public:
	KnnClassifier(std::vector<Sample> &t, unsigned int knn=1);
	virtual ~KnnClassifier();
	virtual unsigned int classify(Sample &s);
protected:
	std::vector<Sample> &trainset;
	unsigned int k;
};

#endif /* KNNCLASSIFIER_H_ */
