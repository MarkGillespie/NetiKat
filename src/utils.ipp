// Generate a random floating point number between fMin and fMax
template <typename T> T fRand(T fMin, T fMax) {
  T f = (T)rand() / (RAND_MAX + 1.0);
  return fMin + f * (fMax - fMin);
}

template <typename T>
std::unordered_map<size_t, T> randMap(size_t n, size_t nEntries, T fMin,
                                      T fMax) {
  std::unordered_map<size_t, T> m;
  for (size_t i = 0; i < nEntries; ++i) {
    m[fRand((size_t)0, nEntries)] += fRand(fMin, fMax);
  }
  return m;
}

// Return a map whose entries sum to 1
template <typename T>
std::unordered_map<size_t, T> randNormalizedMap(size_t n, size_t nEntries) {
  std::unordered_map<size_t, T> m;
  T sum = 0;
  for (size_t i = 0; i < nEntries; ++i) {
    size_t row = fRand((size_t)0, nEntries);
    T val = fRand(0.0, 1.0);
    m[row] += val;
    sum += val;
  }
  for (std::pair<std::size_t, T> entry : m) {
    m[entry.first] = entry.second / sum;
  }
  return m;
}

// Return a map whose entries sum to 1
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
toVec(const std::unordered_map<size_t, T> &m) {
  size_t dim = 0;
  for (std::pair<std::size_t, T> entry : m) {
    dim = std::max(dim, entry.first + 1);
  }
  Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
  for (std::pair<std::size_t, T> entry : m) {
    v(entry.first) = entry.second;
  }

  return v;
}
