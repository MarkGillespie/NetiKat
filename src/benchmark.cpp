#include "TransitionMatrix.h"

#include <stdlib.h> /* rand */

std::vector<size_t> packetType{2, 2, 2, 2};
PacketSet set = PacketSet(packetType);

// Generate a random floating point number between fMin and fMax
double fRand(double fMin, double fMax) {
  double f = (double)rand() / (RAND_MAX + 1.0);
  return fMin + f * (fMax - fMin);
}

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
Eigen::SparseMatrix<double> randomStochastic(size_t n,
                                             size_t entriesPerCol = 24) {
  Eigen::SparseMatrix<double> M(n, n);
  std::vector<Eigen::Triplet<double>> T;
  T.reserve(entriesPerCol * n);
  T.emplace_back(0, 0, 1);

  std::vector<double> entries(entriesPerCol);
  for (size_t col = 1; col < n; ++col) {
    double sum = 0;
    for (size_t i = 0; i < entriesPerCol; ++i) {
      entries[i] = fRand(0, 1);
      sum += entries[i];
    }
    for (size_t i = 0; i < entriesPerCol; ++i) {
      size_t row = rand() % n;
      T.emplace_back(row, col, entries[i] / sum);
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

void benchmark(const PacketSet &set) {
  TransitionMatrix M, p, q;
  std::clock_t start;
  double duration;

  cout << "Generating Random Matrices . . ." << endl;

  // TODO: sample sparse stochastic matrices
  p = randomStochastic(set.matrixDim);
  q = randomStochastic(set.matrixDim);

  cout << "Beginning Test" << endl;

  // Skip
  start = std::clock();
  M = set.skip();
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Skip took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // Drop
  start = std::clock();
  M = set.drop();
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Drop took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // Test
  start = std::clock();
  M = set.test(0, 1);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Test took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // TestSize
  start = std::clock();
  M = set.testSize(0, 1, 2);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "TestSize took " << duration << "s on matrices of size "
       << set.matrixDim << endl
       << endl;

  // Set
  start = std::clock();
  M = set.set(0, 1);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Set took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // Amp
  start = std::clock();
  M = set.amp(p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Amp took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // Seq
  start = std::clock();
  M = set.seq(p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Seq took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;

  // choice
  start = std::clock();
  M = set.choice(0.25, p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Choice took " << duration << "s on matrices of size "
       << set.matrixDim << endl
       << endl;

  // starApprox
  start = std::clock();
  M = set.starApprox(p, 1e-12);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "StarApprox(1e-12) took " << duration << "s on matrices of size "
       << set.matrixDim << endl
       << endl;

  // star
  start = std::clock();
  M = set.star(p);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "Star took " << duration << "s on matrices of size " << set.matrixDim
       << endl
       << endl;
}

void starParameterSweep() {
  cout << "Packet Size \t Matrix Size \t Entries per Column \t Time (s)"
       << endl;

  for (size_t n = 0; n < 16; ++n) {
    PacketSet set = PacketSet(std::vector<size_t>{n});
    for (size_t entries = 2; entries < std::min(set.matrixDim, (size_t)16);
         ++entries) {
      TransitionMatrix p = randomStochastic(set.matrixDim, entries);
      std::clock_t start = std::clock();
      set.star(p);
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      cout << n << "\t" << set.matrixDim << "\t" << entries << "\t" << duration
           << endl;
    }
  }
}

int main() {
  // benchmark(set);
  starParameterSweep();
  return 0;
}
