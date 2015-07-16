/*
 * MyClock.h
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#ifndef MYCLOCK_H_
#define MYCLOCK_H_

#include <chrono>

class MyClock {
public:
	MyClock();
	virtual ~MyClock();
	void start();
	void stop();
	unsigned int getMilliseconds();
protected:
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
};

#endif /* MYCLOCK_H_ */
