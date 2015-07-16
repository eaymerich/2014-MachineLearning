CC=g++
CFLAGS=-std=c++0x -fopenmp -O3
OBJ = gauss.o MyClock.o MyLog.o NgbClassifier.o Sample.o SampleD.o ngb.o

all: ngb.exe

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

ngb.exe: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	del ngb.exe *.o

