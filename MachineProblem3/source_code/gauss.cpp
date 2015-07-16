/*
 * gauss.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: Edward
 */

#include <cmath>
#include <climits>
#define M_PI		3.14159265358979323846

#include "gauss.h"

double gauss(double x, double u, double o2 ){
	return ( 1.0 / sqrt(2.0*M_PI*o2) ) *
			exp( -( ( (x - u)*(x - u) ) / (2.0*o2) ) );
}
