#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <numeric>
#include <omp.h> // <-- uncomment to enable OpenMP routines

using namespace std;

/**
 *
 */
struct generator_config
{
    generator_config(int max) : max(max) {}
    int max;
};

/**
 *
 */
struct generator {
    generator(const generator_config& cfg) : dist(0, cfg.max) {}

    int operator()() {
        return dist(engine);
    }
private:
    minstd_rand engine;
    uniform_int_distribution<int> dist;
};

/**
 *
 */
struct histogram {
    histogram(int count) : data(count) {}

    void add(int i) {
        ++data[i];
    }

    int getSize() {
        return data.size();
    }

    int getData(int i) {
        return data[i];
    }

    void setDataOnIndex(int i, int newVal) {
        data[i] += newVal;
    }

    void print(std::ostream& str) {
        for (size_t i = 0; i < data.size(); ++i) str << i << ":" << data[i] << endl;
        str << "total:" << accumulate(data.begin(), data.end(), 0) << endl;
    }
private:
    vector<int> data;
};

/**
 *
 */
struct worker {
    worker(int repeats_to_do, histogram& h, const generator_config& cfg) : repeats_to_do(repeats_to_do), h(h), cfg(cfg) {}
    int threadNum = omp_get_num_threads();

    void operator()() {
        generator gen(cfg);
        while (repeats_to_do--) {
            int next = gen();
            h.add(next);
        }
    }
private:
    int repeats_to_do;
    histogram& h;
    const generator_config& cfg;
};

//-------------------------------

int main()
{
    int max = 10;
    int repeats_to_do = 500000000;

    generator_config cfg(max);
    histogram h(max+1);

    // replace with omp_get_wtime()
    auto t1 = chrono::high_resolution_clock::now();

    // do it in parallel with OpenMP
    // How and where you split the work depends on
    // what omp construct you want to use.
    // You can do it inside this worker function
    // or here in the main.
    // (uncommented worker function here is just one way to do it)
    worker(repeats_to_do, h, cfg)();

    auto t2 = chrono::high_resolution_clock::now();
    //auto t2 = omp_get_wtime(); // <-- use this time when you switch to OpenMP

    h.print(cout);
    cout << chrono::duration<double>(t2 - t1).count() << endl;
}
