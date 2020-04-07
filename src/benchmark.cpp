#include "netikat.h"

#include <stdlib.h> /* rand, malloc */
#include <sys/mman.h>

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
  size_t count = 0;
  Eigen::Triplet<double> trip;
  for (size_t col = 1; col < n; ++col) {
    if (fmod(col, pow(2, 24)) == 0) {
      cout << "col = " << col << " of " << n << " ( "
           << ((double)col / (double)n) * 100 << "%)" << endl;
    }
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

  M.setFromTriplets(std::begin(T), std::end(T));

  return M;
}

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
Eigen::SparseMatrix<double> randomStochasticLeet(size_t n,
                                                 size_t entriesPerCol = 24) {
  Eigen::SparseMatrix<double> M(n, n);

  void *mem = malloc(n * entriesPerCol * (2 * sizeof(int) + sizeof(double)));
  if (mem == NULL) {
    cout << "Failed to allocate memory" << endl;
    exit(1);
  }

  int *intArr = (int *)mem;
  double *doubleArr = (double *)mem;

  intArr[0] = 0;
  intArr[1] = 0;
  doubleArr[1] = 1;

  double *entries = new double[entriesPerCol];
  size_t count = 0;
  Eigen::Triplet<double> trip;
  for (size_t col = 1; col < n; ++col) {
    if (fmod(col, pow(2, 24)) == 0) {
      cout << "col = " << col << " of " << n << " ( "
           << ((double)col / (double)n) * 100 << "%)" << endl;
    }
    double sum = 0;
    for (size_t i = 0; i < entriesPerCol; ++i) {
      entries[i] = fRand(0, 1);
      sum += entries[i];
    }
    for (size_t i = 0; i < entriesPerCol; ++i) {
      size_t row = rand() % n;
      intArr[4 * count] = row;
      intArr[4 * count + 1] = col;
      doubleArr[2 * count + 1] = entries[i] / sum;
      count++;
    }
  }
  delete[] entries;

  Eigen::Triplet<double> *T = (Eigen::Triplet<double> *)mem;
  M.setFromTriplets(T, T + entriesPerCol * n);
  free(mem);

  return M;
}

void benchmarkMatrixGeneration(size_t n) {
  TransitionMatrix p;
  std::clock_t start;
  double duration;

  cout << "Generating Random Matrix with array" << endl;
  start = std::clock();
  p = randomStochasticLeet(n);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "generation took " << duration << "s on matrices of size " << n
       << endl
       << endl;

  cout << "Generating Random Matrix with vector" << endl;
  start = std::clock();
  p = randomStochastic(n);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "generation took " << duration << "s on matrices of size " << n
       << endl
       << endl;
}

void benchmark(const NetiKAT &set) {
  TransitionMatrix M, p, q;
  std::clock_t start;
  double duration;

  cout << "Generating Random Matrices . . ." << endl;

  // TODO: sample sparse stochastic matrices
  p = randomStochasticLeet(set.matrixDim);
  q = randomStochasticLeet(set.matrixDim);

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
  M = set.starApprox(p, 1e-8);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "StarApprox(1e-12) took " << duration << "s on matrices of size "
       << set.matrixDim << endl
       << endl;

  // // star
  // start = std::clock();
  // M = set.star(p);
  // duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  // cout << "Star took " << duration << "s on matrices of size " <<
  // set.matrixDim
  //      << endl
  //      << endl;
}

void starParameterSweep() {
  cout << "Packet Size \t Matrix Size \t Entries per Column \t Time "
          "(s)\tIterations"
       << endl;

  for (size_t n = 1; n < 64; n *= 2) {
    NetiKAT set = NetiKAT(std::vector<size_t>{n}, 3);
    for (size_t entries = 2; entries < std::min(set.matrixDim, (size_t)16);
         entries *= 2) {
      TransitionMatrix p = randomStochastic(set.matrixDim, entries);
      std::clock_t start = std::clock();
      // set.star(p);
      size_t iterations;
      set.starApprox(p, 1e-8, iterations);
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      cout << n << "\t" << set.matrixDim << "\t" << entries << "\t" << duration
           << "\t" << iterations << endl;
    }
  }
}

int main() {

  std::vector<size_t> packetType{64};
  NetiKAT neti = NetiKAT(packetType, 4);
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
