/*
 * KnnClassifier.h
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#ifndef KNNCLASSIFIER_H_
#define KNNCLASSIFIER_H_

#include <vector>
#include "SampleD.h"

class KnnClassifier {
public:
	KnnClassifier(std::vector<SampleD> &t, unsigned int knn=1);
	virtual ~KnnClassifier();
	virtual unsigned int classify(SampleD &s);
protected:
	std::vector<SampleD> &trainset;
	unsigned int k;
};

#endif /* KNNCLASSIFIER_H_ */
