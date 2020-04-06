#include "TransitionMatrix.h"

#include <stdlib.h> /* rand, malloc */
#include <sys/mman.h>

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

  // Eigen::Triplet<double> *T = new Eigen::Triplet<double>[entriesPerCol * n];
  // cout << "Triplet size: " << sizeof(Eigen::Triplet<double>) << endl;
  // Eigen::Triplet<double> *T = (Eigen::Triplet<double> *)malloc(
  //     entriesPerCol * n * sizeof(Eigen::Triplet<double>));

  // void *mem = malloc(pow(2, 64) * sizeof(Eigen::Triplet<double>));
  // void *mem = malloc(pow(2, 5000) * sizeof(int));

  // if (mem == NULL) {
  //   cout << "Failed to allocate enough memory" << endl;
  //   exit(1);
  // } else {
  //   cout << "Succeeded in allocating memory" << endl;
  // }
  // Eigen::Triplet<double> *T = (Eigen::Triplet<double> *)mem;

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
      // trip = Eigen::Triplet<double>(row, col, entries[i] / sum);
      // T[count++] = trip;
    }
  }

  M.setFromTriplets(std::begin(T), std::end(T));
  // M.setFromTriplets(T, T + entriesPerCol * n);

  // delete[] T;
  // free(T);
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
  cout << "Packet Size \t Matrix Size \t Entries per Column \t Time "
          "(s)\tIterations"
       << endl;

  for (size_t n = 1; n < 64; n *= 2) {
    PacketSet set = PacketSet(std::vector<size_t>{n});
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

bool testAlloc(size_t n) {
  char *array = (char *)malloc(n * sizeof(char));
  if (array == NULL)
    return false;

  for (size_t i = 0; i < n; ++i) {
    array[i] = 12;
  }

  free(array);
  return true;
}

bool testMMapAlloc(size_t n) {
  size_t bufferSize = n * sizeof(char);
  void *addr = mmap(NULL, bufferSize, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED)
    return false;

  char *array = (char *)addr;

  for (size_t i = 0; i < n; ++i) {
    array[i] = 12;
  }

  munmap(addr, bufferSize);
  return true;
}

bool testTripletVectorAlloc(size_t n) {
  std::vector<Eigen::Triplet<double>> T;
  T.reserve(n);
  for (size_t iT = 0; iT < n; ++iT) {
    T.emplace_back(0, 0, 1);
  }
  return true;
}

bool testSneakyTripletAlloc(size_t n) {

  int nBytes = n * (2 * sizeof(int) + sizeof(double));

  char *array = (char *)malloc(nBytes);

  if (array == NULL)
    return false;

  int *intArray = (int *)array;
  double *doubleArray = (double *)array;

  for (size_t iT = 0; iT < n; ++iT) {
    intArray[4 * iT] = iT;          // row
    intArray[4 * iT + 1] = iT;      // col
    doubleArray[2 * iT + 1] = 3.14; // value;
  }

  Eigen::Triplet<double> *T = (Eigen::Triplet<double> *)array;
  Eigen::SparseMatrix<double> M(n, n);
  M.setFromTriplets(T, T + n);
  free(array);

  cout << M.coeffRef(0, 0) << "\t" << M.coeffRef(1, 0);

  return true;
}

size_t maxMemoryAllocation() {

  cout << 2 * sizeof(int) + sizeof(double) << "\t" << sizeof(char) << endl;
  size_t maxMem = pow(2, 26);
  while (testMMapAlloc(maxMem) && maxMem * 2 > maxMem) {
    cout << "Successfully allocated " << maxMem << " bytes\t" << log2(maxMem)
         << endl;
    maxMem *= 2;
  }

  cout << endl;
  maxMem = pow(2, 26);
  // while (testAlloc(maxMem) && maxMem * 2 > maxMem) {
  // while (testTripletVectorAlloc(maxMem) && maxMem * 2 > maxMem) {
  while (testSneakyTripletAlloc(maxMem) && maxMem * 2 > maxMem) {
    cout << "Successfully allocated " << maxMem << " Triplets\t" << log2(maxMem)
         << endl;
    maxMem *= 2;
  }

  return maxMem;
}

int main() {
  // benchmark(set);
  // starParameterSweep();

  // size_t n = 27;
  // size_t entries = 4;
  // PacketSet set = PacketSet(std::vector<size_t>{n});
  // cout << "Constructing Matrix" << endl;
  // TransitionMatrix p = randomStochastic(set.matrixDim, entries);
  // size_t iterations;
  // cout << "Computing star" << endl;
  // set.starApprox(p, 1e-8, iterations);

  maxMemoryAllocation();

  return 0;
}
