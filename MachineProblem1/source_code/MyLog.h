/*
 * MyLog.h
 *
 *  Created on: Aug 30, 2014
 *      Author: Edward
 */

#ifndef MYLOG_H_
#define MYLOG_H_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// Adapted from: http://stackoverflow.com/questions/2212776/overload-handling-of-stdendl

class MyLog: public std::ostream
{
    // Write a stream buffer that prefixes each line with Plop
    class MyStreamBuf: public std::stringbuf
    {
        std::ostream&   output;
        std::ostream&   foutput;
        public:
            MyStreamBuf(std::ostream& str,std::ostream& str2)
                :output(str),foutput(str2)
            {}

        // When we sync the stream with the output.
        // 1) Output Plop then the buffer
        // 2) Reset the buffer
        // 3) flush the actual output stream we are using.
        virtual int sync ( )
        {
        	foutput << str();
            output << str();
            str("");
            foutput.flush();
            output.flush();
            return 0;
        }
    };

    std::ofstream file;
    // My Stream just uses a version of my special buffer
    MyStreamBuf buffer;
    public:
    	MyLog(std::string filename)
            :std::ostream(&buffer)
            ,file(filename, std::ios_base::app),buffer(std::cout, file)
        {}
    	virtual ~MyLog() {file.close();}
};

#endif /* MYLOG_H_ */
