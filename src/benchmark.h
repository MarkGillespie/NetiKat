#include "netikat.h"
#include "utils.h"

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
template <typename T>
TransitionMatrix<T> randomTransitionMatrix(size_t n, size_t entriesPerCol = 24);

template <typename T>
void benchmark(const NetiKAT<T> &neti, size_t entriesPerCol = 24,
               bool verbose = false);

template <typename T> void starParameterSweep();

#include "benchmark.ipp"
