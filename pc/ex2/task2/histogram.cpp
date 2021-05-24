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
        #pragma omp atomic
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
struct workerStatic {
    workerStatic(int repeats_to_do, histogram& h, const generator_config& cfg) : repeats_to_do(repeats_to_do), h(h), cfg(cfg) {}
	int threadNum = omp_get_num_threads();

	void operator()() {
        generator gen(cfg);
        #pragma omp parallel
        {
            histogram hist = h;
            #pragma omp for nowait firstprivate(gen) schedule(static)
            for (int i =0; i < repeats_to_do; ++i) {
                int next = gen();
                hist.add(next);
            };

            #pragma omp critical
            {
                for (int i = 0; i < hist.getSize(); ++i) {
                    h.setDataOnIndex(i, hist.getData(i));
                }
            };
	    }
	}
private:
	int repeats_to_do;
	histogram& h;
	const generator_config& cfg;
};

/**
 *
 */
struct workerDynamic {
    workerDynamic(int repeats_to_do, histogram& h, const generator_config& cfg, int chunkSize = 1) : repeats_to_do(repeats_to_do), h(h), cfg(cfg), chunkSize(chunkSize) {}
    int threadNum = omp_get_num_threads();

    void operator()() {
        generator gen(cfg);
        #pragma omp parallel
        {
            histogram hist = h;
            #pragma omp for nowait firstprivate(gen) schedule(dynamic, chunkSize)
            for (int i =0; i < repeats_to_do; ++i) {
                int next = gen();
                hist.add(next);
            };

            #pragma omp critical
            {
                for (int i = 0; i < hist.getSize(); ++i) {
                    h.setDataOnIndex(i, hist.getData(i));
                }
            };
        }
    }
private:
    int repeats_to_do;
    int chunkSize;
    histogram& h;
    const generator_config& cfg;
};

/**
 *
 */
struct workerGuided {
    workerGuided(int repeats_to_do, histogram& h, const generator_config& cfg) : repeats_to_do(repeats_to_do), h(h), cfg(cfg) {}
    int threadNum = omp_get_num_threads();

    void operator()() {
        generator gen(cfg);
        #pragma omp parallel
        {
            histogram hist = h;
            #pragma omp for nowait firstprivate(gen) schedule(guided)
            for (int i =0; i < repeats_to_do; ++i) {
                int next = gen();
                hist.add(next);
            };

            #pragma omp critical
            {
                for (int i = 0; i < hist.getSize(); ++i) {
                    h.setDataOnIndex(i, hist.getData(i));
                }
            };
        }
    }
private:
    int repeats_to_do;
    histogram& h;
    const generator_config& cfg;
};

enum WorkerType {
    Static, Dynamic1, Dynamic100, Guided
};

/**
 *
 * @param type
 */
void executeHistogram(WorkerType type) {
    int max = 10;
    int repeats_to_do = 500000000;

    std::vector<int> nrOfThreads = {1,2,4,8,16,32};

    generator_config cfg(max);

    for (const auto& nr : nrOfThreads) {
        histogram h(max+1);
        omp_set_num_threads(nr);
        auto t1 = omp_get_wtime();

        switch(type) {
            case WorkerType::Dynamic1:
                std::cout << "Dynamic Scheduling (chunk size 1) " << std::endl;
                workerDynamic(repeats_to_do, h, cfg, 1)();
                break;
            case WorkerType::Dynamic100:
                std::cout << "Dynamic Scheduling (chunk size 100) " << std::endl;
                workerDynamic(repeats_to_do, h, cfg, 100)();
                break;
            case WorkerType::Guided:
                std::cout << "Guided Scheduling " << std::endl;
                workerGuided(repeats_to_do, h, cfg)();
                break;
            case WorkerType::Static:
                std::cout << "Static Scheduling " << std::endl;
                workerStatic(repeats_to_do, h, cfg)();
        }

        auto t2 = omp_get_wtime();
        std::cout << "Nr. of threads: " << nr << std::endl;
        h.print(cout);
        cout << chrono::duration<double>(t2 - t1).count() << endl;
    }
}


/**
 *
 * @return
 */
int main()
{
    executeHistogram(WorkerType::Dynamic1);
    executeHistogram(WorkerType::Dynamic100);
    executeHistogram(WorkerType::Static);
    executeHistogram(WorkerType::Guided);
}
