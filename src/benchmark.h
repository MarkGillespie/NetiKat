#include "netikat.h"

#include <stdlib.h> /* rand, malloc */
#include <sys/mman.h>

// Generate a random floating point number between fMin and fMax
template <typename T> T fRand(T fMin, T fMax);

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
template <typename T>
Eigen::SparseMatrix<T> randomStochastic(size_t n, size_t entriesPerCol = 24);

template <typename T>
void benchmark(const NetiKAT<T> &neti, size_t entriesPerCol = 24,
               bool verbose = false, bool runFullStar = false);

template <typename T> void starParameterSweep();

#include "benchmark.ipp"
