#include "benchmark.h"

int main() {

  std::vector<double> sizes{1e5, 1e6,   5e6, 1e7,   2.5e7,
                            5e7, 7.5e7, 1e8, 1.5e8, 2e8};
  for (double i : sizes) {
    if (i > std::numeric_limits<size_t>::max()) {
      cout << "ERROR: " << i << " is to big to be represented as a size_t"
           << endl;
    }
    std::vector<size_t> packetType{(size_t)i};
    NetiKAT<double> neti(packetType, 1);
    benchmark(neti);
  }

  // starParameterSweep();

  // size_t n = 32;
  // size_t entries = 4;
  // NetiKAT set = NetiKAT(std::vector<size_t>{n}, 4);
  // cout << "Constructing Matrix" << endl;
  // TransitionMatrix p = randomStochastic(set.matrixDim, entries);
  // size_t iterations;
  // cout << "Computing star" << endl;
  // set.starApprox(p, 1e-8, iterations);

  return 0;
}
