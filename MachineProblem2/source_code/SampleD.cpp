/*
 * SampleD.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Edward
 */

#include <iostream>
using namespace std;
#include <cmath>

#include "gauss.h"
#include "MyLog.h"

#include "SampleD.h"

const double SampleD::epsilon = 0.12;

MyLog mlog("data.txt");

SampleD::SampleD(const SampleD &sd) :
	data(sd.data){

}

SampleD::SampleD(unsigned int size_d) :
	data(size_d) {

}

SampleD::SampleD(const Sample& s) :
	data(s.data.size()) {

	for(unsigned int i = 0; i < s.data.size(); i++){
		data[i] = s.data[i];
	}
}

SampleD::~SampleD() {
	data.clear();
}

void SampleD::squared(){
	for(unsigned int i = 0; i < data.size(); i++){
		data[i] *= data[i];
	}
}

void SampleD::operator=(const SampleD &sd){
	data = move(std::vector<double>(sd.data));
}

void SampleD::operator=(SampleD &&sd){
	data = move(sd.data);
}

void SampleD::operator=(const Sample &s){
	if(data.size() != s.data.size()){
		data = move(std::vector<double>(s.data.size()));
	}

	for(unsigned int i = 0; i < data.size(); i++){
		data[i] = s.data[i];
	}
}

void SampleD::operator=(const double val){
	for(unsigned int i = 0; i < data.size(); i++){
		data[i] = val;
	}
}

void SampleD::operator+=(const SampleD &sd){
	if(data.size() == sd.data.size()){
		for(unsigned int i = 0; i < data.size(); i++){
			data[i] += sd.data[i];
		}
	}
}

void SampleD::operator+=(const Sample &s){
	if(data.size() == s.data.size()){
		for(unsigned int i = 0; i < data.size(); i++){
			data[i] += s.data[i];
		}
	}
}

void SampleD::operator+=(const double val){
	for(unsigned int i = 0; i < data.size(); i++){
		data[i] += val;
	}
}

void SampleD::operator*=(const double val){
	for(unsigned int i = 0; i < data.size(); i++){
		data[i] *= val;
	}
}

void SampleD::sum_and_square(Sample &s, SampleD &sum, SampleD &square){
	if (s.data.size() == sum.data.size() && s.data.size() == square.data.size()){
		for(unsigned int i = 0; i < s.data.size(); i++){
			double d = static_cast<double>( s.data[i] );
			sum.data[i] += d;
			square.data[i] += (d*d);
		}
	}else{
		std::cerr << "ERROR: SampleD::sum_and_square(): sizes don't match." << std::endl;
	}
}

void SampleD::mean_and_variance(const unsigned int n, const SampleD &sum, const SampleD &square, SampleD &mean, SampleD &variance){
	if(sum.data.size() == square.data.size() && sum.data.size() == mean.data.size() && sum.data.size() == variance.data.size()) {
		double inv_n = 1.0 / static_cast<double>( n );
		for(unsigned int i = 0; i < sum.data.size(); i++){
			mean.data[i] = inv_n * sum.data[i];
			variance.data[i] = inv_n * (square.data[i] - (inv_n*sum.data[i]*sum.data[i]) ) + epsilon;
		}
	}else{
		std::cerr << "ERROR: SampleD::mean_and_variance(): sizes don't match." << std::endl;
	}
}

double SampleD::likelihood(const Sample &x, const SampleD &mean, const SampleD &variance){
	double sum = 0.0;
	if(x.data.size() == mean.data.size() && x.data.size() == variance.data.size()){
		for(unsigned int i = 0; i < x.data.size(); i++){
			double xx = (double)x.data[i];
			double u = mean.data[i];
			double o2 = variance.data[i];
			double g = gauss( xx, u, o2);
			double l = log(g);
			sum += l;
/*
			mlog << "i=" << i;
			mlog << " xx=" << xx;
			mlog << " u=" << u;
			mlog << " o2=" << o2;
			mlog << " g=" << g;
			mlog << " l=" << l;
			mlog << " sum=" << sum << endl;*/
		}
	}else{
		std::cerr << "ERROR: SampleD::posteriori(): sizes don't match." << std::endl;
	}
	return sum;
}
