/*
 * FDA.h
 *
 *  Created on: Oct 13, 2014
 *      Author: Edward
 */

#ifndef FDA_H_
#define FDA_H_

#define EIGEN_MPL2_ONLY 1
#include <vector>
//#include "Eigen/src/Core/Matrix.h"
#include "Eigen/Dense"
#include "MyLog.h"

class FDA {
public:
	FDA(MyLog& _mlog);
	virtual ~FDA();
	void train(const Eigen::MatrixXd& x, const std::vector<uint8_t> label);
	void transform(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);
private:
	void reduce(const Eigen::MatrixXd& x, Eigen::MatrixXd& rx);
	void fast_reduce(const Eigen::MatrixXd& x, Eigen::MatrixXd& rx);
	MyLog& mlog;
	Eigen::MatrixXd w;
	std::vector<unsigned int> nzd;
};

#endif /* FDA_H_ */
