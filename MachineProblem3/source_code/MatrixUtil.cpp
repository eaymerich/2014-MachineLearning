/*
 * MatrixUtil.cpp
 *
 *  Created on: Oct 14, 2014
 *      Author: Edward
 */

#include <iostream>
#include <fstream>
#include "MatrixUtil.h"

using namespace std;
using namespace Eigen;

void writeMatrix(string filename, MatrixXd &m){
	ofstream file(filename, ios_base::out | ios_base::trunc | ios_base::binary);
	if(!file){
		cerr << "Error opening file \"" << filename << "\"." << endl;
		//mlog << endl << "ERROR: Error opening file \"" << filename << "\"." << endl;
		exit(EXIT_FAILURE);
	}

	uint32_t rows = m.rows();
	uint32_t cols = m.cols();
	file.write(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
	file.write(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

	double* data = m.data();
	file.write(reinterpret_cast<char*>(data), sizeof(double)*rows*cols);

	file.close();
}

bool readMatrix(string filename, MatrixXd &m){
	ifstream file(filename, ios_base::in | ios_base::binary);
	if(!file){
		cerr << "Error opening file \"" << filename << "\"." << endl;
		//mlog << "Error opening file \"" << filename << "\"." << endl;
		return false;
	}

	uint32_t rows, cols;
	file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

	m = move( MatrixXd(rows, cols) );

	double *data = m.data();
	file.read(reinterpret_cast<char*>(data), sizeof(double)*rows*cols);

	file.close();

	return true;
}
