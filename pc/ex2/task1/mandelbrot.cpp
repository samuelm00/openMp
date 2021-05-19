#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <omp.h>

using namespace std;

typedef vector<int> Array1D;
typedef vector<Array1D> Array2D;
typedef vector<Array2D> Image;

// Test if point c belong to the Mandelbrot Set
bool mandelbrot(complex<double> c, vector<int>& pixel) {
    int max_iteration = 1000, iteration = 0;
    complex<double> z(0, 0);

    while ( abs(z) <= 4 && (iteration < max_iteration) ) {
        z = z * z + c;
        iteration++;
    };

    if (iteration != max_iteration) {
        // calculate these values to colorize the output
        // e.g: if r-g-b all have the same value, 
        //      you will have a gray color of some intensity
        //      so you can adjusting acording to some number - (thread-id maybe?)
        pixel = {255, 255, 255}; // ouside -> white 255,255,255
        return false;
    }


    pixel = {omp_get_thread_num()* 30, omp_get_thread_num()* 30, omp_get_thread_num()* 30}; // ouside -> white 255,255,255
    return true;
}

int main(int argc, char **argv) {
    const int width = 1200, height = 1200;

    int i, j, pixels_inside=0;

    // Image data structure: 
    // - for each pixel we need red, green, and blue values (0-255)
    // - we use 3 different matrices for corresponding channels
    const int channels = 3; // red, green, blue
    Image image(channels, Array2D(height, Array1D(width)));

    // pixel to be passed to the mandelbrot function
    vector<int> pixel = {0,0,0}; // red,green,blue (each range 0-255)
    complex<double> c;

	//auto t1 = chrono::high_resolution_clock::now();
	auto t1 = omp_get_wtime(); // <-- use this time when you switch to OpenMP

    #pragma omp parallel for collapse(2) private(c) firstprivate(pixel, height, width)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            c = complex<double>( 2.0*((double)j/width-0.75), ((double)i/height-0.5)*2.0);

                if ( mandelbrot(c, pixel) ) {
                    #pragma omp critical
                    {
                        pixels_inside++;
                    }
                };

            for (int ch = 0; ch < channels; ch++)
                image[ch][i][j] = pixel[ch];

        }
    }
    
	// auto t2 = chrono::high_resolution_clock::now();
	auto t2 = omp_get_wtime(); // <-- use this time when you switch to OpenMP
    
    // save image
    std::ofstream ofs("mandelbrot.ppm", std::ofstream::out);
    ofs << "P3" << std::endl;
    ofs << width << " " << height << std::endl;
    ofs << 255 << std::endl;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            ofs << " " << image[0][i][j] << " " << image[1][i][j] << " " << image[2][i][j] << std::endl;
        }
    }
    ofs.close();

    cout << "Total pixels inside: " << pixels_inside << endl;
    cout << "Execution time (without disk I/O)): " << chrono::duration<double>(t2 - t1).count() << endl;

    return 0;
}

/*
 * No OMP:
 * Total pixels inside: 543518
 * Execution time (without disk I/O)): 23.494
 */