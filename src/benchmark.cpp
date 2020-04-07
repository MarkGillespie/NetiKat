#include "benchmark.h"

int main() {

  NetiKAT<double, 64> neti(4);
  cout << "Matrix size: " << neti.matrixDim << endl;
  benchmark(neti);
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
