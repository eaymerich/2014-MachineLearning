/*
 * MyClock.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: Edward
 */

#include "MyClock.h"

MyClock::MyClock() {
}

MyClock::~MyClock() {
}

void MyClock::start(){
	begin = std::chrono::steady_clock::now();
}

void MyClock::stop(){
	end = std::chrono::steady_clock::now();
}

unsigned int MyClock::getMilliseconds(){
	return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
}
