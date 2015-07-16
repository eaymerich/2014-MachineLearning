/*
 * FDA.cpp
 *
 *  Created on: Oct 13, 2014
 *      Author: Edward
 */

#define EIGEN_MPL2_ONLY 1

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
//#include "Eigen/Dense"
#include "FDA.h"
#include "MyClock.h"

using namespace std;
using namespace Eigen;

FDA::FDA(MyLog& _mlog) : mlog(_mlog) {

}

FDA::~FDA() {

}

void FDA::fast_reduce(const MatrixXd& x, MatrixXd& rx){
	// Copy non-empty dimensions into reduced matrix
	rx = move(MatrixXd(nzd.size(),x.cols()));
	for(unsigned int i = 0; i < nzd.size(); i++){
		rx.row(i) = x.row(nzd[i]);
	}
}

// 0. Find out witch dimensions have all 0 values and eliminate them.
void FDA::reduce(const MatrixXd& x, MatrixXd& rx){
	unsigned int d = x.rows();
	unsigned int total_n = x.cols();

	// Sum out all data points
	VectorXd z = VectorXd::Zero(d);
	for(unsigned int i = 0; i < total_n; i++){
		z += x.col(i);
	}
	//cout << "zero:" << endl << z.transpose();
	//cout << endl;

	// Count how many dimensions aren't empty.
	//unsigned int dd = 0;
	for(unsigned int i = 0; i < d; i++){
		if(z(i) > 0){
			//++dd;
			nzd.push_back(i);
		}
	}
	// Copy non-empty dimensions into reduced matrix
	rx = move(MatrixXd(nzd.size(),total_n));
	for(unsigned int i = 0; i < nzd.size(); i++){
		rx.row(i) = x.row(nzd[i]);
	}

	/*
	unsigned int ddi = 0;
	for(unsigned int i = 0; i < d; i++){
		if(z(i) > 0){
			rx.row(ddi) = x.row(i);
			ddi++;
		}
	}*/
}

void FDA::train(const MatrixXd& x, const vector<uint8_t> label){
	MyClock clock;

	//cout << "X size: " << x.rows() << "x" << x.cols() << endl;
	// 0. Reduce input matrix so onlu non empty rows remains
	MatrixXd rx;
	reduce(x,rx);
	//cout << "rX size: " << rx.rows() << "x" << rx.cols() << endl;

	unsigned int k = 10;
	unsigned int d = rx.rows();
	unsigned int total_n = rx.cols();


	// 1. Count how many elements are in each class and compute mean vectors
	mlog << "Computing means..."; flush(mlog);
	clock.start();
	unsigned int N[k]{0};
	vector<VectorXd> m{k, VectorXd::Zero(d)};
	for(unsigned int i = 0; i < total_n; i++){
		m[label[i]] += rx.col(i);
		++N[label[i]];
	}
	for(unsigned int i = 0; i < k; i++){
		m[i] /= (double)N[i];
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

	// 2. Compute Scatter Matrices
	// 2.a Compute Between-class Scatter Matrix S_B
	mlog << "Computing Sb..."; flush(mlog);
	clock.start();
	MatrixXd sb = MatrixXd::Zero(d,d);
	for(unsigned int i=0; i < k; i++){
		for(unsigned int j=i+1; j < k; j++){
			if(i != j){
				VectorXd tmp = m[i] - m[j];
				//sb += ((m[i] - m[j]) * (m[i] - m[j]).transpose());
				sb += (tmp * tmp.transpose());
			}
		}
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;


	// 2.b Compute Within-class Scatter Matrix S_W
	mlog << "Computing Sw..."; flush(mlog);
	MatrixXd sw = MatrixXd::Zero(d,d);
	vector<MatrixXd> s{k, MatrixXd::Zero(d,d)};
	VectorXd tmp;
	clock.start();
	int th_id, nthreads;
	unsigned int n;
	#pragma omp parallel private(n,th_id,nthreads,tmp)
	{
		th_id = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		for(n=0; n < total_n; n++){
			if(label[n] % nthreads == th_id){
				tmp = rx.col(n) - m[label[n]];
				s[label[n]] += (tmp * tmp.transpose());
			}
		}
	}
	for(unsigned int i = 0; i < k; i++){
		sw += (1.0/(double)N[i]) * s[i];
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

//	ofstream out("sb.txt");
//	out << sb;
//	out.close();

	//ofstream out("sw.txt");
	//out << sw;
	//out.close();


	// 2.b Compute Within-class Scatter Matrix S_W
	/*
	mlog << "Computing SW..."; flush(mlog);
	clock.start();
	//for(unsigned int n=0; n < total_n; n++){
	//	rx.col(n) -= m[label[n]];
	//}
	MatrixXd sw1 = MatrixXd::Zero(d,d);
	s = move(vector<MatrixXd>{k, MatrixXd::Zero(d,d)});
	for(unsigned int n=0; n < total_n; n++){
		unsigned int i = label[n];
		//s[i] += rx.col(n)*rx.col(n).transpose();
		VectorXd tmp = rx.col(n) - m[i];
		s[i] += (tmp * tmp.transpose());
		//s[i] += (x.col(n) - m[i]) * (x.col(n) - m[i]).transpose();
	}
	for(unsigned int i = 0; i < k; i++){
		sw1 += (1.0/(double)N[i]) * s[i];
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	*/

	// Compare sw abd sw1
//	for(unsigned int i=0; i < d; i++){
//		for(unsigned int j=0; j < d; j++){
//			if(sw(j,i) != sw1(j,i)){
//				cerr << "Wrong value at " << i << "," << j << endl;
//			}
//		}
//	}
//	cout << "Finish comparing SW and Sw1." << endl;

	/*
	//--- Slow > 5.5min
	MatrixXd sw = MatrixXd::Zero(d,d);
	MatrixXd si = MatrixXd::Zero(d,d);
	VectorXd tmp;
	unsigned int i;
	#pragma omp parallel for schedule(static) private(i,si,tmp) reduction(+:sw)
	for(i = 0; i < k; i++){
		for(unsigned int n=0; n < total_n; n++){
			if(i == label[n]){
				tmp = x.col(n) - m[i];
				si += tmp * tmp.transpose();
			}
		}
		sw += si;
	}
	*/
/*
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
*/

	// 3. Solve generalized eigenvalue problem for S_W^{-1}S_B
	mlog << "Computing eigenvectors..."; flush(mlog);
	clock.start();
	//GeneralizedEigenSolver<MatrixXd> ges;
	//ges.compute(sb,sw);
	//mlog << "Eigenvalues:" << endl << ges.betas().transpose();
	//mlog << "Eigenvalues:" << endl << ges.eigenvalues().transpose();

	//MatrixXd tmpE = sw.inverse()*sb;
	//EigenSolver<MatrixXd> eig;
	//eig.compute(tmpE);
	//mlog << "Eigenvalues: " << endl << eig.eigenvalues().transpose();

	GeneralizedSelfAdjointEigenSolver<MatrixXd>	eigen;
	eigen.compute(sb, sw);

	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
	//mlog << "Eigenvalues: " << endl << eigen.eigenvalues().transpose();
	//mlog << endl;
	//mlog << endl;

	// 4. Calculate matrix W using the top k-1 eigenvectors
	//    Note: eigen returns the top eigenvalues at the end
	mlog << "Computing W..."; flush(mlog);
	clock.start();
	w = move(MatrixXd(d,k-1));
	MatrixXd eigenvectors = eigen.eigenvectors();
	for(unsigned int i = 0; i < k-1; i++){
		w.col(i) = eigenvectors.col(eigenvectors.cols()-1-i);
	}
	clock.stop();
	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;

	// 5. Transform original dataset to new space
//	mlog << "Computing Y..."; flush(mlog);
//	clock.start();
//	y = w.transpose()*rx;
//	clock.stop();
//	mlog << "OK. Time: " << clock.getMilliseconds() << "ms." << endl;
}

void FDA::transform(const Eigen::MatrixXd& x, Eigen::MatrixXd& y){
	MatrixXd xx;
	fast_reduce(x,xx);
	y = w.transpose()*xx;
}
