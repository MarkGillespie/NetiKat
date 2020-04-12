// Generate a random stochastic matrix of size nxn with entriesPerCol nonzero
// entries per column. Default value of entriesPerCol is 24
template <typename T>
TransitionMatrix<T> randomTransitionMatrix(size_t n, size_t entriesPerCol) {
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

  TransitionMatrix<T> f = [=](Distribution<T> p) -> Distribution<T> {
    // TODO: you really shouldn't do this
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);

    // Iterate over an unordered_map using range based for loop
    for (std::pair<std::size_t, T> entry : p) {
      v(entry.first) = entry.second;
    }

    Eigen::VectorXd Mv = M * v;
    std::unordered_map<size_t, T> pOut;
    for (size_t i = 0; i < n; ++i) {
      if (Mv(i) > 1e-8) {
        pOut[i] = Mv(i);
      }
    }
    return pOut;
  };

  return f;
}

template <typename T>
void benchmark(const NetiKAT<T> &neti, size_t entriesPerCol, bool verbose) {
  TransitionMatrix<T> M, a, b;
  Distribution<T> v;
  std::clock_t start;
  double duration;

  if (verbose) {
    cout << "Generating Random Matrices of size " << neti.matrixDim << " . . ."
         << endl;
  }

  a = randomTransitionMatrix<T>(neti.matrixDim, entriesPerCol);
  b = randomTransitionMatrix<T>(neti.matrixDim, entriesPerCol);
  v = randMap(neti.matrixDim, entriesPerCol, 0.0, 1.0);
  neti.normalize(v);

  if (verbose) {
    cout << "Beginning Test" << endl;
    cout << "MatrixDim\tSkip\tDrop\tTest\tTestSize\tSet\tAmp\tSeq\tChoice\tStar"
            "Approx"
         << endl;
  }

  cout << neti.matrixDim << "\t";

  // Skip
  start = std::clock();
  M = neti.skip();
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Drop
  start = std::clock();
  M = neti.drop();
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Test
  start = std::clock();
  M = neti.test(0, 1);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // TestSize
  start = std::clock();
  M = neti.testSize(0, 1, 2);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Set
  start = std::clock();
  M = neti.set(0, 1);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Amp
  start = std::clock();
  M = neti.amp(a, b);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Seq
  start = std::clock();
  M = neti.seq(a, b);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // Choice
  start = std::clock();
  M = neti.choice(0.25, a, b);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << duration << "\t";

  // starApprox
  start = std::clock();
  M = neti.starApprox(a, 1e-8);
  M(v);
  duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
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
      TransitionMatrix<T> p = randomTransitionMatrix(neti.matrixDim, entries);
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
