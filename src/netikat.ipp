template <typename T>
NetiKAT<T>::NetiKAT(const PacketType &type_, size_t maxNumPackets_)
    : packetType(type_), maxNumPackets(maxNumPackets_) {
  possiblePackets = 1;
  for (size_t fieldSize : packetType) {
    possiblePackets *= fieldSize;
  }

  numNetiKATsOfSizeLessThan.reserve(maxNumPackets + 1);
  numNetiKATsOfSizeLessThan.push_back(0);
  for (size_t i = 1; i <= maxNumPackets; ++i) {
    // The number of packet sets of size less than i is the number of packet
    // sets of size less than i-1 plus the number of packet sets of size exactly
    // i-1
    numNetiKATsOfSizeLessThan.push_back(numNetiKATsOfSizeLessThan[i - 1] +
                                        binom(possiblePackets, i - 1));
  }
  matrixDim = numNetiKATsOfSizeLessThan[maxNumPackets] +
              binom(possiblePackets, maxNumPackets);

  // TODO: use less memory?
  // stores all n choose k for 0 <= n <= possiblePackets,
  // 0 <= k <= maxNumPackets
  binomialCoefficients.reserve(pow(possiblePackets + 1, 2));
  for (size_t n = 0; n <= possiblePackets; ++n) {
    for (size_t k = 0; k <= maxNumPackets; ++k) {
      if (k <= n) {
        binomialCoefficients.push_back(binom(n, k));
      } else {
        binomialCoefficients.push_back(1234);
      }
    }
  }
}

template <typename T>
size_t NetiKAT<T>::binomialCoefficient(size_t n, size_t k) const {
  // if (k > n || k > maxNumPackets) {
  //   throw std::invalid_argument("In binom(n, k), k must be less than or equal
  //   "
  //                               "to n and maxNumPackets. But k is " +
  //                               std::to_string(k) + ", n is " +
  //                               std::to_string(n) + ", and maxNumPackets is "
  //                               + std::to_string(maxNumPackets));
  // } else if (n > possiblePackets) {
  //   throw std::invalid_argument("I only precomputed binomials up to "
  //                               "possiblePackets. But possiblePackets is " +
  //                               std::to_string(possiblePackets) + " and n is
  //                               " + std::to_string(n));

  // } else {
  return binomialCoefficients[k + n * (maxNumPackets + 1)];
  // }
}

template <typename T> size_t NetiKAT<T>::packetIndex(const Packet &p) const {
  size_t idx = 0;

  size_t offset = 1;
  for (size_t iF = 0; iF < p.size(); ++iF) {
    idx += p[iF] * offset;
    offset *= packetType[iF];
  }

  return idx;
}

template <typename T> Packet NetiKAT<T>::packetFromIndex(size_t idx) const {
  Packet p;

  for (size_t iF = 0; iF < packetType.size(); ++iF) {
    size_t field = idx % packetType[iF];
    p.push_back(field);
    idx -= field;
    idx /= packetType[iF];
  }
  return p;
}

// The indexing scheme for packet sets is as follows:
// First of all, packet sets are ordered by cardinality. Within each collection
// of packet sets with the same cardinality, the packet sets are ordered
// lexicographically
// Packet sets which are too big index to 0
template <typename T> size_t NetiKAT<T>::index(const PacketSet &packets) const {

  if (packets.size() > maxNumPackets) {
    return 0;
  }

  // TODO: will iterating over the set iterate in sorted order?
  std::vector<size_t> presentIndices;
  presentIndices.reserve(packets.size());
  for (size_t pIdx : packets) {
    presentIndices.push_back(pIdx);
  }
  std::sort(std::begin(presentIndices), std::end(presentIndices));

  // Index of packet set among sets of same size
  size_t peerIndex = 0;

  // I need to ensure that prevIdx + 1 = 0 the first time I use prevIdx, so I
  // set prevIdx to -1. Since prevIdx is a size_t this underflows. But that
  // doesn't matter since -1 is still the additive inverse of 1 even when
  // working with unsigned ints
  size_t prevIdx = -1;
  for (size_t iIdx = 0; iIdx < packets.size(); ++iIdx) {
    size_t packetIdx = presentIndices[iIdx];
    for (size_t iSkip = prevIdx + 1; iSkip < packetIdx; ++iSkip) {
      // If there's a zero in position iSkip, you're behind all of the packets
      // that have a 1 there instead, and thus have (packets.size() - iIdx) ones
      // in the last few positions
      peerIndex += binomialCoefficient(possiblePackets - iSkip - 1,
                                       packets.size() - iIdx - 1);
    }
    prevIdx = packetIdx;
  }

  return peerIndex + numNetiKATsOfSizeLessThan[packets.size()];
}

template <typename T>
Eigen::VectorXd NetiKAT<T>::toVec(const PacketSet &packets) const {
  size_t pIdx = index(packets);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(matrixDim);
  v(pIdx) = 1;
  return v;
}

template <typename T>
PacketSet NetiKAT<T>::packetSetFromIndex(size_t idx) const {
  PacketSet packets;

  size_t nPackets = 0;
  while (nPackets + 1 <= maxNumPackets &&
         numNetiKATsOfSizeLessThan[nPackets + 1] <= idx) {
    nPackets++;
  }

  size_t nPeers = binomialCoefficient(possiblePackets, nPackets);

  size_t peerIndex = idx - numNetiKATsOfSizeLessThan[nPackets];
  size_t reconstructedIndex = 0;
  size_t packetsLeft = nPackets;
  for (size_t iP = 0; iP < possiblePackets - packetsLeft && packetsLeft != 0;
       ++iP) {
    size_t nextTerm =
        binomialCoefficient(possiblePackets - iP - 1, packetsLeft);
    if (peerIndex < nPeers - nextTerm) {
      packets.insert(iP);
      packetsLeft -= 1;
      nPeers -= nextTerm;
    }
  }
  for (size_t iP = possiblePackets - packetsLeft; iP < possiblePackets; ++iP) {
    packets.insert(iP);
  }

  return packets;
}

template <typename T> TransitionMatrix<T> NetiKAT<T>::drop() const {
  Eigen::SparseMatrix<T> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<T>> trip;

  for (size_t i = 0; i < matrixDim; ++i) {
    trip.emplace_back(0, i, 1);
  }

  M.setFromTriplets(trip.begin(), trip.end());
  return M;
}

template <typename T> TransitionMatrix<T> NetiKAT<T>::skip() const {
  return speye(matrixDim);
}

// B[fieldIndex = fieldValue]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::test(size_t fieldIndex,
                                     size_t fieldValue) const {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> trip;

  for (size_t i = 0; i < matrixDim; ++i) {
    PacketSet packetsIn = packetSetFromIndex(i);
    PacketSet packetsOut;
    for (size_t iP : packetsIn) {
      Packet p = packetFromIndex(iP);
      if (p[fieldIndex] == fieldValue) {
        packetsOut.insert(packetIndex(p));
      }
    }
    trip.emplace_back(index(packetsOut), i, 1);
  }

  M.setFromTriplets(trip.begin(), trip.end());
  return M;
}

// B[#fieldIndex = fieldValue : n]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::testSize(size_t fieldIndex, size_t fieldValue,
                                         size_t n) const {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> trip;

  for (size_t i = 0; i < matrixDim; ++i) {
    PacketSet packetsIn = packetSetFromIndex(i);
    size_t nPacketsOut = 0;
    for (size_t iP : packetsIn) {
      Packet p = packetFromIndex(iP);
      if (p[fieldIndex] == fieldValue) {
        nPacketsOut += 1;
      }
    }

    if (nPacketsOut == n) {
      trip.emplace_back(i, i, 1);
    } else {
      trip.emplace_back(i, 0, 1);
    }
  }

  M.setFromTriplets(trip.begin(), trip.end());
  return M;
}

// B[fieldIndex <- fieldValue]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::set(size_t fieldIndex,
                                    size_t fieldValue) const {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> trip;

  for (size_t i = 0; i < matrixDim; ++i) {
    PacketSet packetsIn = packetSetFromIndex(i);
    PacketSet packetsOut;
    for (size_t iP : packetsIn) {
      Packet p = packetFromIndex(iP);
      p[fieldIndex] = fieldValue;
      packetsOut.insert(packetIndex(p));
    }
    trip.emplace_back(index(packetsOut), i, 1);
  }

  M.setFromTriplets(trip.begin(), trip.end());
  return M;
}

template <typename T>
size_t NetiKAT<T>::packetSetUnion(size_t a, size_t b) const {
  PacketSet p = packetSetFromIndex(a);
  PacketSet q = packetSetFromIndex(b);
  p.insert(std::begin(q), std::end(q));
  return index(p);
}

// B[p & q]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::amp(TransitionMatrix<T> p,
                                    TransitionMatrix<T> q) const {

  // Sort nonzero entries by "a" value
  // Map each a to {(b, val)}
  // (a is col, b is row)
  std::vector<std::vector<std::pair<size_t, double>>> pTriplets, qTriplets;
  pTriplets.resize(matrixDim);
  qTriplets.resize(matrixDim);
  for (int kP = 0; kP < p.outerSize(); ++kP) {
    for (Eigen::SparseMatrix<double>::InnerIterator itP(p, kP); itP; ++itP) {
      pTriplets[itP.col()].emplace_back(itP.row(), itP.value());
    }
  }
  for (int kQ = 0; kQ < q.outerSize(); ++kQ) {
    for (Eigen::SparseMatrix<double>::InnerIterator itQ(q, kQ); itQ; ++itQ) {
      qTriplets[itQ.col()].emplace_back(itQ.row(), itQ.value());
    }
  }

  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> trip;

  size_t bP, bQ;
  double valP, valQ;
  for (size_t a = 0; a < matrixDim; ++a) {
    for (std::pair<size_t, double> pPair : pTriplets[a]) {
      std::tie(bP, valP) = pPair;
      for (std::pair<size_t, double> qPair : qTriplets[a]) {
        std::tie(bQ, valQ) = qPair;

        if (valP * valQ > 1e-12) {
          size_t packetUnion = packetSetUnion(bP, bQ);
          trip.emplace_back(packetUnion, a, valP * valQ);
        }
      }
    }
  }

  M.setFromTriplets(trip.begin(), trip.end());
  normalize(M);
  return M;
}

// B[p;q]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::seq(TransitionMatrix<T> p,
                                    TransitionMatrix<T> q) const {
  return p * q;
}

// B[p \oplus_r q]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::choice(T r, TransitionMatrix<T> p,
                                       TransitionMatrix<T> q) const {
  return r * p + (1 - r) * q;
}

template <typename T> size_t NetiKAT<T>::bigIndex(size_t i, size_t j) const {
  return i + matrixDim * j;
}

template <typename T>
std::pair<size_t, size_t> NetiKAT<T>::bigUnindex(size_t i) const {
  return std::make_pair(i % matrixDim, floor(i / matrixDim));
}

// TODO: clean up, write more tests
// TODO: normalize columns after pruning small values?
// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::star(TransitionMatrix<T> p) const {

  Eigen::SparseMatrix<double> S(matrixDim * matrixDim, matrixDim * matrixDim);
  Eigen::SparseMatrix<double> U(matrixDim * matrixDim, matrixDim * matrixDim);

  std::vector<Eigen::Triplet<double>> Ts, Tu;

  auto nonzeros = [](Eigen::VectorXd v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (v(i) > 1e-8) {
        out += std::to_string(i) + " ";
      }
    }
    return out;
  };

  std::vector<std::vector<size_t>> ancestors;
  for (size_t a = 0; a < matrixDim * matrixDim; ++a) {
    ancestors.push_back(std::vector<size_t>());
  }

  for (size_t a = 0; a < matrixDim; ++a) {
    for (size_t b = 0; b < matrixDim; ++b) {
      size_t col = bigIndex(a, b);

      Eigen::VectorXd aVec = Eigen::VectorXd::Zero(matrixDim);
      aVec(a) = 1;
      Eigen::VectorXd aPrimeVec = p * aVec;

      for (size_t aPrime = 0; aPrime < matrixDim; ++aPrime) {
        if (abs(aPrimeVec(aPrime)) < 1e-8) {
          continue;
        }
        // Take union of sets
        size_t bPrime = packetSetUnion(a, b);
        size_t row = bigIndex(aPrime, bPrime);
        Ts.emplace_back(row, col, aPrimeVec(aPrime));
        ancestors[row].push_back(col);
      }
    }
  }

  Vector<bool> isSaturated = Vector<bool>::Ones(matrixDim * matrixDim);
  std::function<void(size_t)> markUnsaturated = [ancestors, &isSaturated,
                                                 &markUnsaturated](size_t ind) {
    isSaturated(ind) = false;
    for (size_t a : ancestors[ind]) {
      if (isSaturated(a)) {
        markUnsaturated(a);
      }
    }
  };

  S.setFromTriplets(Ts.begin(), Ts.end());
  for (int k = 0; k < S.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
      size_t a, b, aPrime, bPrime;
      std::tie(a, b) = bigUnindex(it.col());
      std::tie(aPrime, bPrime) = bigUnindex(it.row());

      if (b != bPrime) {
        markUnsaturated(it.col());
      }
    }
  }
  ancestors.clear();

  for (size_t a = 0; a < matrixDim; ++a) {
    for (size_t b = 0; b < matrixDim; ++b) {
      size_t col = bigIndex(a, b);
      if (isSaturated(col)) {
        size_t row = bigIndex(0, b);
        Tu.emplace_back(row, col, 1);
      } else {
        size_t row = bigIndex(a, b);
        Tu.emplace_back(row, col, 1);
      }
    }
  }

  Vector<bool> isAbsorbing = Vector<bool>::Zero(matrixDim * matrixDim);
  for (size_t b = 0; b < matrixDim; ++b) {
    isAbsorbing(bigIndex(0, b)) = true;
  }

  U.setFromTriplets(Tu.begin(), Tu.end());
  Ts.clear();
  Tu.clear();

  Eigen::SparseMatrix<double> SU = S * U;
  S.resize(0, 0);
  U.resize(0, 0);

  BlockDecompositionResult<double> decomp =
      blockDecomposeSquare(SU, isAbsorbing);

  Eigen::SparseMatrix<double> Rt = decomp.AB.transpose();
  const Eigen::SparseMatrix<double> &Q = decomp.BB;

  Eigen::SparseMatrix<double> IminusQ = (speye(Q.rows()) - Q).transpose();
  Eigen::SparseMatrix<double> X = solveSquare(IminusQ, Rt).transpose();
  Rt.resize(0, 0);

  Eigen::SparseMatrix<double> limit =
      reassembleMatrix(decomp, speye(decomp.AA.rows()), X);

  // Check that limit is really limit
  Eigen::SparseMatrix<double> SUlimit = SU * limit;
  double err = (SUlimit - limit).norm();
  if (err > 1e-12) {
    cerr << "Error: limit wrong somehow" << endl;
    exit(1);
  }

  Eigen::SparseMatrix<double> smallLimit(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> trip;

  for (int k = 0; k < limit.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(limit, k); it; ++it) {
      if (abs(it.value()) < 1e-8) {
        continue;
      }
      size_t a, b, aPrime, bPrime;
      std::tie(a, b) = bigUnindex(it.col());
      std::tie(aPrime, bPrime) = bigUnindex(it.row());
      if (aPrime == 0 && b == 0) {
        trip.emplace_back(bPrime, a, it.value());
      }
    }
  }

  smallLimit.setFromTriplets(trip.begin(), trip.end());
  return smallLimit;
}

// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::starApprox(TransitionMatrix<T> p, T tol) const {
  size_t temp;
  return starApprox(p, tol, temp);
}

// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::starApprox(TransitionMatrix<T> p, T tol,
                                           size_t &iterationsNeeded) const {
  TransitionMatrix<T> oldS = skip();
  TransitionMatrix<T> s = amp(skip(), seq(oldS, p));
  iterationsNeeded = 1;

  while ((s - oldS).norm() > tol) {
    oldS = s;
    s = amp(skip(), seq(s, p));
    normalize(s);
    iterationsNeeded++;
  }

  return s;
}

// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::dumbStarApprox(TransitionMatrix<T> p,
                                               size_t iter) const {
  TransitionMatrix<T> s = skip();

  for (size_t i = 0; i < iter; ++i) {
    s = amp(skip(), seq(s, p));
    normalize(s);
  }

  return s;
}

template <typename T> void NetiKAT<T>::normalize(TransitionMatrix<T> &M) const {
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(M.rows());
  Eigen::VectorXd colSums = ones.transpose() * M;
  for (int k = 0; k < M.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
      it.valueRef() /= colSums(it.col());
    }
  }
}
