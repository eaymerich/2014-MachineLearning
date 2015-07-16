/*
 * PCA.h
 *
 *  Created on: Oct 14, 2014
 *      Author: Edward
 */

#ifndef PCA_H_
#define PCA_H_

#define EIGEN_MPL2_ONLY 1

#include "Eigen/Dense"
#include "MyLog.h"

class PCA {
public:
	PCA(MyLog& _mlog);
	virtual ~PCA();
	void train(const Eigen::MatrixXd& x);
	void transform(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);
private:
	MyLog& mlog;
	Eigen::VectorXd m;
	Eigen::MatrixXd w;
};

#endif /* PCA_H_ */
