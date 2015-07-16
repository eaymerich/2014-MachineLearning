/*
 * MatrixUtil.h
 *
 *  Created on: Oct 14, 2014
 *      Author: Edward
 */

#ifndef MATRIXUTIL_H_
#define MATRIXUTIL_H_

#include "Eigen\Dense"

bool readMatrix(std::string filename, Eigen::MatrixXd &m);
void writeMatrix(std::string filename, Eigen::MatrixXd &m);

#endif /* MATRIXUTIL_H_ */
