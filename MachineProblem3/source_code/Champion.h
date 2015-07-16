/*
 * Champion.h
 *
 *  Created on: Aug 28, 2014
 *      Author: Edward
 */

#ifndef CHAMPION_H_
#define CHAMPION_H_

#include <vector>
using namespace std;

class Champion {
public:
	Champion(unsigned int n=1);
	Champion(const Champion &c);
	virtual ~Champion();
	void testContestant(unsigned int p, double v);
	unsigned int getChampion(unsigned int n);
	unsigned int size();
	void combine(Champion &other);
protected:
	vector<unsigned int> pos;
	vector<double> value;
	static int count;
};

#endif /* CHAMPION_H_ */
