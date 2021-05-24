histogram: histogram.o
	g++ histogram.o -fopenmp -O2 -o histogram

histogram.o: histogram.cpp
	g++ -c histogram.cpp -fopenmp -O2

clean:
	rm *.o