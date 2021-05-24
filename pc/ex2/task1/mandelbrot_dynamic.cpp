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
bool mandelbrotDynamicCheck(complex<double> c, vector<int>& pixel) {
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


    pixel = {omp_get_thread_num()*5, omp_get_thread_num()*5, omp_get_thread_num()*5}; // ouside -> white 255,255,255
    return true;
}

void mandelBrotDynamic() {
    const int width = 1200, height = 1200;

    // Image data structure: 
    // - for each pixel we need red, green, and blue values (0-255)
    // - we use 3 different matrices for corresponding channels
    const int channels = 3; // red, green, blue
    Image image(channels, Array2D(height, Array1D(width)));

    std::vector<int> nrOfThreads = {1,2,4,8,16,32};

    for (const auto& nr: nrOfThreads) {
        omp_set_num_threads(nr);

        int i, j, pixels_inside=0;
        vector<int> pixel = {0,0,0};
        complex<double> c;

        auto t1 = omp_get_wtime();

        #pragma omp parallel for schedule(dynamic, 12) collapse(2) private(c) firstprivate(pixel, height, width)
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                c = complex<double>( 2.0*((double)j/width-0.75), ((double)i/height-0.5)*2.0);

                if ( mandelbrotDynamicCheck(c, pixel) ) {
                    #pragma omp atomic
                    pixels_inside++;
                };

                for (int ch = 0; ch < channels; ch++)
                    image[ch][i][j] = pixel[ch];

            }
        }

        // auto t2 = chrono::high_resolution_clock::now();
        auto t2 = omp_get_wtime(); // <-- use this time when you switch to OpenMP

        // save image
        std::ofstream ofs("mandelbrot_dynamic.ppm", std::ofstream::out);
        ofs << "P3" << std::endl;
        ofs << width << " " << height << std::endl;
        ofs << 255 << std::endl;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                ofs << " " << image[0][i][j] << " " << image[1][i][j] << " " << image[2][i][j] << std::endl;
            }
        }
        ofs.close();

        cout << "Dynamic scheduling" << endl;
        cout << "Thread Nr: " << nr << endl;
        cout << "Total pixels inside: " << pixels_inside << endl;
        cout << "Execution time (without disk I/O)): " << chrono::duration<double>(t2 - t1).count() << endl;

    }
}

/*
 * No OMP:
 * Total pixels inside: 543518
 * Execution time (without disk I/O)): 23.494
 */