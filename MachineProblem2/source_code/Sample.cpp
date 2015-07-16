/*
 * Sample.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: Edward
 */

#include "Sample.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

Sample::Sample() : label(0), width(0), height(0) {

}

Sample::Sample(uint32_t width_t, uint32_t height_t, uint8_t lbl)
	: label(lbl), width(width_t), height(height_t), data(width*height){

}

Sample::~Sample() {
	data.clear();
}

/*
 * Calculates Euclidean distance to another sample.
 */
double Sample::distance(Sample &s){
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
	//return d;
}

void read_uint32(ifstream &in, uint32_t &n){
	in.read((char*)&n, sizeof(n));
	n = __builtin_bswap32(n);
}

void Sample::loadSamples(string sample_filename, string label_filename, vector<Sample> &v){

	// IDX data
	uint32_t label_magic_number;
	uint32_t labels;

	uint32_t sample_magic_number;
	uint32_t samples;
	uint32_t sample_height;
	uint32_t sample_width;

	ifstream in_label(label_filename, ios_base::in | ios_base::binary);
	read_uint32(in_label,label_magic_number);
	if(label_magic_number != 2049){
		cerr << "ERROR: label file doesn't start with magic number." << endl;
		return;
	}
	read_uint32(in_label,labels);

	//cout << "Label Magic Number= " << label_magic_number << endl;
	//cout << "# of labels= " << labels << endl;

	ifstream in_sample(sample_filename, ios_base::in | ios_base::binary);
	read_uint32(in_sample,sample_magic_number);
	if(sample_magic_number != 2051){
		cerr << "ERROR: sample file doesn't start with magic number." << endl;
		return;
	}
	read_uint32(in_sample,samples);
	read_uint32(in_sample,sample_height);
	read_uint32(in_sample,sample_width);

	//cout << "Sample Magic Number= " << sample_magic_number << endl;
	//cout << "# of samples= " << samples << endl;
	//cout << "Sample height= " << sample_height << endl;
	//cout << "Sample width= " << sample_width << endl;

	if(samples != labels){
		cerr << "ERROR: number of samples doesn't match number of labels" << endl;
		return;
	}

	// Read samples
	for(uint32_t s=0; s < samples; s++){
		// Read sample label
		uint8_t lbl;
		in_label.read((char*)&(lbl), sizeof(lbl));

		Sample sample(sample_width, sample_height, lbl);
		vector<uint8_t>buff(sample_width*sample_height);

		// Read sample vector
		if(!in_sample.read((char*)&(buff[0]),sample_height*sample_width)){
			cerr << "Fail reading data" << endl;
		}
		for(unsigned int i = 0; i < sample.data.size(); i++){
			sample.data[i] = ((double)buff[i]) / 255.0;
		}

		v.push_back(sample);
	}

	in_sample.close();
	in_label.close();
}

ostream& operator<<(ostream& os, const Sample& sample){
	uint32_t i=0;
	os << "label=" << (int)sample.label << endl;
	for(uint32_t y=0; y < sample.height; y++){
		for(uint32_t x=0; x < sample.width; x++){
			if(sample.data[i++] == 0.0){
				os << ".";
			}else{
				os << "#";
			}
		}
		os << endl;
	}

	return os;
}
