mandelbrot: mandelbrot_dynamic.o mandelbrot_guided.o mandelbrot_static.o
	g++ mandelbrot_static.o -fopenmp -O2 -o mandelbrot

mandelbrot_static.o: mandelbrot_static.cpp mandelbrot_dynamic.cpp mandelbrot_guided.cpp
	g++ -c mandelbrot_static.cpp mandelbrot_dynamic.cpp mandelbrot_guided.cpp -fopenmp -O2

clean:
	rm *.o