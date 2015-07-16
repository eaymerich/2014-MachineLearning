/*
 * Champion.cpp
 *
 *  Created on: Aug 28, 2014
 *      Author: Edward
 */

#include <iostream>
#include <limits>
using namespace std;

#include "Champion.h"

int Champion::count = 0;

Champion::Champion(unsigned int n)
	: pos(n,0), value(n, numeric_limits<double>::max()) {
	//++count;
	//cout << "+There are " << count << " Champion objects." << endl;
}

Champion::Champion(const Champion &c)
	: pos(c.pos), value(c.value){
	//++count;
	//cout << "++There are " << count << " Champion objects." << endl;
}

Champion::~Champion() {
	//--count;
	//cout << "-There are " << count << " Champion objects." << endl;
}

void Champion::testContestant(unsigned int p, double v){
	if(v < value[0]){
		// Contestant is worthy, enter the competition
		value[0] = v;
		pos[0] = p;

		// Compete against remaining champions
		unsigned int p_temp;
		double v_temp;
		for(unsigned int i = 1; i < value.size() && value[i-1] < value[i]; i++){
			// Exchange values
			v_temp = value[i];
			value[i] = value[i-1];
			value[i-1] = v_temp;

			// Exchange positions
			p_temp = pos[i];
			pos[i] = pos[i-1];
			pos[i-1] = p_temp;
		}
	}
}

unsigned int Champion::getChampion(unsigned int n){
	if(n < pos.size()){
		return pos[pos.size()-1 - n];
	}
	return 0;
}

unsigned int Champion::size(){
	return pos.size();
}

void Champion::combine(Champion &other){
	for(unsigned int i = 0; i < other.size(); i++){
		this->testContestant(other.pos[i], other.value[i]);
	}
}
