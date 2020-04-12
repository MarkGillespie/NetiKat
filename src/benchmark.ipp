// Generate a random floating point number between fMin and fMax
template <typename T> T fRand(T fMin, T fMax) {
  T f = (T)rand() / (RAND_MAX + 1.0);
  return fMin + f * (fMax - fMin);
}

// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
template <typename T>
Eigen::SparseMatrix<T> randomStochastic(size_t n, size_t entriesPerCol) {
  Eigen::SparseMatrix<T> M(n, n);
  std::vector<Eigen::Triplet<double>> trip;
  trip.reserve(entriesPerCol * n);
  trip.emplace_back(0, 0, 1);

  std::vector<T> entries(entriesPerCol);
  size_t count = 0;
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
      trip.emplace_back(row, col, entries[i] / sum);
    }
  }

  M.setFromTriplets(std::begin(trip), std::end(trip));

  return M;
}

template <typename T>
void benchmark(const NetiKAT<T> &neti, size_t entriesPerCol, bool verbose,
               bool runFullStar) {
  TransitionMatrix<T> p, q;
  std::clock_t start;
  double duration;

  if (verbose) {
    cout << "Generating Random Matrices of size " << neti.matrixDim << " . . ."
         << endl;
  }

  p = randomStochastic<T>(neti.matrixDim, entriesPerCol);
  q = randomStochastic<T>(neti.matrixDim, entriesPerCol);

  if (verbose) {
    cout << "Beginning Test" << endl;
    cout << "MatrixDim\tSkip\tDrop\tTest\tTestSize\tSet\tAmp\tSeq\tChoice\tStar"
            "Approx"
         << endl;
  }

  cout << neti.matrixDim << "\t";

  // Skip
  start = std::clock();
  neti.skip();
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Drop
  start = std::clock();
  neti.drop();
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Test
  start = std::clock();
  neti.test(0, 1);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // TestSize
  start = std::clock();
  neti.testSize(0, 1, 2);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Set
  start = std::clock();
  neti.set(0, 1);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Amp
  start = std::clock();
  neti.amp(p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Seq
  start = std::clock();
  neti.seq(p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Choice
  start = std::clock();
  neti.choice(0.25, p, q);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // starApprox
  start = std::clock();
  neti.starApprox(p, 1e-8);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

  // star
  if (runFullStar) {
    cout << duration << "\t";
    start = std::clock();
    neti.star(p);
    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  }

  cout << duration << endl;
}

template <typename T> void starParameterSweep() {
  cout << "Packet Size \t Matrix Size \t Entries per Column \t Time "
          "(s)\tIterations"
       << endl;

  for (size_t n = 1; n < 64; n *= 2) {
    NetiKAT<T> neti = NetiKAT<T>(std::vector<size_t>{n}, 3);
    for (size_t entries = 2; entries < std::min(neti.matrixDim, (size_t)16);
         entries *= 2) {
      TransitionMatrix<T> p = randomStochastic(neti.matrixDim, entries);
      std::clock_t start = std::clock();
      // neti.star(p);
      size_t iterations;
      neti.starApprox(p, 1e-8, iterations);
      double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
      cout << n << "\t" << neti.matrixDim << "\t" << entries << "\t" << duration
           << "\t" << iterations << endl;
    }
  }
}
