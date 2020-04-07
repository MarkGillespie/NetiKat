#include "netikat.h"

#include <stdlib.h> /* rand, malloc */
#include <sys/mman.h>

// Generate a random floating point number between fMin and fMax
template <typename T, size_t... ns> T fRand(T fMin, T fMax);

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
template <typename T, size_t... ns>
Eigen::SparseMatrix<T> randomStochastic(size_t n, size_t entriesPerCol = 24);

template <typename T, size_t... ns>
void benchmark(const NetiKAT<T, ns...> &neti, bool runFullStar = false);

template <typename T, size_t... ns> void starParameterSweep();

#include "benchmark.ipp"
