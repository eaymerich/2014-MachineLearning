/*
 * NgbClassifier.h
 *
 *  Created on: Sep 6, 2014
 *      Author: Edward
 */

#ifndef NGBCLASSIFIER_H_
#define NGBCLASSIFIER_H_

#include <vector>
#include "Sample.h"
#include "SampleD.h"

class NgbClassifier {
public:
	NgbClassifier(std::vector<Sample> &t, double a = 0.0);
	virtual ~NgbClassifier();
	virtual void train();
	virtual unsigned int classify(Sample &s);
	void incrementVariance();
protected:
	std::vector<Sample> &trainset;
	double alpha;
	std::vector<double> prior;
	std::vector<SampleD> mean;
	std::vector<SampleD> variance;
};

#endif /* NGBCLASSIFIER_H_ */
