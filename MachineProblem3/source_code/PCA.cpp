/*
 * PCA.cpp
 *
 *  Created on: Oct 14, 2014
 *      Author: Edward
 */

#include <iostream>
#include "Eigen/Dense"
#include "MyClock.h"
#include "PCA.h"

using namespace std;
using namespace Eigen;

PCA::PCA(MyLog& _mlog) : mlog(_mlog) {

}

PCA::~PCA() {

}

void PCA::train(const MatrixXd& xx){
	MyClock clock;
	MatrixXd x = xx;

	// 1. Create matrix X = dxN, with d=dimensions and N=number of data points.
	// Note already given, no need to do this here.

	// 2. Subtract mean m from each column vector in X.
	mlog << "Calculating and subtracting mean..."; flush(mlog);
	clock.start();
	m = VectorXd::Constant(x.rows(),0.0);
	for(unsigned int i = 0; i < x.cols(); i++){
		m += x.col(i);
	}
	m /= (double)x.cols();
	for(unsigned int i = 0; i < x.cols(); i++){
		x.col(i) -= m;
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

	// 3. Compute covariance matrix sigma = XX^T
	MatrixXd sigma;
	mlog << "Computing sigma..."; flush(mlog);
	clock.start();
	sigma = x * x.transpose();
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." <<endl;

	// 4. Find eigenvalues and eigenvectors of sigma.
	mlog << "Computing eigenvalues and eigenvectors..."; flush(mlog);
	clock.start();
	SelfAdjointEigenSolver<MatrixXd> eigensolver(sigma);
	clock.stop();
	if(eigensolver.info() != Success){
		cerr << "Error solving for eigenvalues!" << endl;
		exit(EXIT_FAILURE);
	}
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." <<endl;

	// 5. Set k principal components to the eigenvectors with the largest k values.
	VectorXd eigenvalues = eigensolver.eigenvalues();
	double total_e = 0.0, acu_e = 0.0, percent = 0.0;
	for(unsigned int i=0; i < eigenvalues.rows(); i++){
		total_e += eigenvalues(i);
	}
	unsigned int pca = 0;
	while(percent < 95.0){
		acu_e += eigenvalues(eigenvalues.rows()-1-pca);
		percent = acu_e / total_e * 100.0;
		pca++;
	}
	mlog << "Using " << pca << " principal components, with " << percent << "% of total eigenvalues." << endl;

	mlog << "Calculating W..."; flush(mlog);
	clock.start();
	MatrixXd eigenvectors = eigensolver.eigenvectors();
	//cout << "eigenvectors: " << eigenvectors.rows() << "x" << eigenvectors.cols() << endl;
	w = move(MatrixXd(x.rows(), pca));
	//cout << "w: " << w.rows() << "x" << w.cols() << endl;
	for(unsigned int i = 0; i < pca; i++){
		w.col(i) = eigenvectors.col(eigenvectors.cols()-1-i);
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;


	// 6. Transform original dataset to new space
	/*
	mlog << "Computing Y..."; flush(mlog);
	clock.start();
	y = w.transpose()*x;
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	*/

/*
	mlog << "Eigen acumulators: " << endl;
	VectorXd eigenvalues = eigensolver.eigenvalues();
	double total_e = 0.0, acu_e = 0.0;
	for(unsigned int i=0; i < eigenvalues.rows(); i++){
		total_e += eigenvalues(i);
	}
	int nn = 1;
	for(unsigned int i=eigenvalues.rows()-1; i >= 1; i--){
		acu_e += eigenvalues(i);
		mlog << "comps=" << nn << " " << acu_e / total_e * 100.0 << "% i=" << i << endl;
		nn++;
	}
	*/

	/*
	vector<unsigned int> pcs{1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,784};
	for(auto pc : pcs){
		MatrixXd x_hat;
		string x_hat_filename = "../MNIST/pca/x_hat_train_" + to_string(pc) + ".dat";
		mlog << "Calculating PCA representation for " << pc << " principal components..."; flush(mlog);
		clock.start();
		x_hat = eigenvectors.block(0,0,eigenvectors.rows(),pc).transpose() * x;
		clock.stop();
		mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
		saveMatrix(x_hat_filename, x_hat);
	}*/
}

void PCA::transform(const MatrixXd& x, MatrixXd& y){
	MatrixXd xx(x.rows(),x.cols());
	for(unsigned int i = 0; i < xx.cols(); i++){
		xx.col(i) = x.col(i) - m;
	}
	y = w.transpose()*xx;
}
