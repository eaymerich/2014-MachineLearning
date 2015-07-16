/*
 * NgbClassifier.h
 *
 *  Created on: Sep 6, 2014
 *      Author: Edward
 */

#ifndef NGBCLASSIFIER_H_
#define NGBCLASSIFIER_H_

#include <vector>
#include "SampleD.h"

class NgbClassifier {
public:
	NgbClassifier(std::vector<SampleD> &t, double a = 0.0);
	virtual ~NgbClassifier();
	void train();
	unsigned int classify(SampleD &s);
	void incrementVariance();
protected:
	std::vector<SampleD> &trainset;
	double alpha;
	std::vector<double> prior;
	std::vector<SampleD> mean;
	std::vector<SampleD> variance;
};

#endif /* NGBCLASSIFIER_H_ */
