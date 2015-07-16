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

//const double SampleD::epsilon = 0.12;
//const double SampleD::epsilon = 0.00001;
const double SampleD::epsilon = 0.000001;

SampleD::SampleD(const SampleD &sd) :
	label(sd.label), data(sd.data)  {

}

SampleD::SampleD(unsigned int size_d, uint8_t lbl) :
	label(lbl), data(size_d) {

}

SampleD::~SampleD() {
	data.clear();
}

unsigned int SampleD::size(){
	return data.size();
}

/*
void SampleD::squared(){
	for(unsigned int i = 0; i < data.size(); i++){
		data[i] *= data[i];
	}
}
*/

void SampleD::operator=(const SampleD &sd){
	data = move(std::vector<double>(sd.data));
}

void SampleD::operator=(SampleD &&sd){
	data = move(sd.data);
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

double SampleD::distance(const SampleD &s){
	double d = 0.0;
	double t;
	if(data.size() != s.data.size()){
		return -1.0;
	}
	for (vector<uint8_t>::size_type i = 0; i < data.size(); i++){
		t = (double)data[i] - (double)s.data[i];
		d += t*t;
	}
	return sqrt(d);
}

void SampleD::sum_and_square(SampleD &s, SampleD &sum, SampleD &square){
	if (s.data.size() == sum.data.size() && s.data.size() == square.data.size()){
		for(unsigned int i = 0; i < s.data.size(); i++){
			double d = s.data[i];
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

double SampleD::likelihood(const SampleD &x, const SampleD &mean, const SampleD &variance){
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

void sampled_read_uint32(ifstream &in, uint32_t &n){
	in.read((char*)&n, sizeof(n));
	n = __builtin_bswap32(n);
}

void SampleD::loadSamplesFromMatrix(std::string sample_filename, std::string label_filename, std::vector<SampleD> &v){

	// IDX data
	uint32_t label_magic_number;
	uint32_t labels;

	uint32_t samples;
	uint32_t sample_size;
	//uint32_t sample_width;

	ifstream in_label(label_filename, ios_base::in | ios_base::binary);
	sampled_read_uint32(in_label,label_magic_number);
	if(label_magic_number != 2049){
		cerr << "ERROR: label file doesn't start with magic number." << endl;
		return;
	}
	sampled_read_uint32(in_label,labels);

	//cout << "Label Magic Number= " << label_magic_number << endl;
	//cout << "# of labels= " << labels << endl;

	ifstream in_sample(sample_filename, ios_base::in | ios_base::binary);
	in_sample.read(reinterpret_cast<char*>(&sample_size), sizeof(uint32_t));
	in_sample.read(reinterpret_cast<char*>(&samples), sizeof(uint32_t));

	//cout << "# of samples= " << samples << endl;
	//cout << "Sample size= " << sample_size << endl;

	if(samples != labels){
		cerr << "ERROR: number of samples doesn't match number of labels" << endl;
		return;
	}

	// Read samples
	v.clear();
	for(uint32_t s=0; s < samples; s++){
		// Read sample label
		uint8_t lbl;
		in_label.read((char*)&(lbl), sizeof(lbl));

		SampleD sample(sample_size, lbl);

		// Read sample vector
		if(!in_sample.read((char*)&(sample.data[0]),sizeof(double)*sample_size)){
			cerr << "Fail reading data" << endl;
		}

		v.push_back(sample);
	}

	in_sample.close();
	in_label.close();
}

ostream& operator<<(ostream& os, const SampleD& sample){
	os << "label=" << (int)sample.label << " [";
	for(uint32_t i=0; i < sample.data.size()-1; i++){
		os << sample.data[i] << ",";
	}
	os << sample.data[sample.data.size()-1] << "]" << endl;
	return os;
}
