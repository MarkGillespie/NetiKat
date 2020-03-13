#include "TransitionMatrix.h"

PacketSet::PacketSet(const PacketType &type_) : packetType(type_) {
  possiblePackets = 1;
  for (size_t fieldSize : packetType) {
    possiblePackets *= fieldSize;
  }
  matrixDim = pow(2, possiblePackets);
}

size_t PacketSet::packetIndex(const Packet &p) {
  size_t idx = 0;

  size_t offset = 1;
  for (size_t iF = 0; iF < p.size(); ++iF) {
    idx += p[iF] * offset;
    offset *= packetType[iF];
  }

  return idx;
}

Packet PacketSet::packetFromIndex(size_t idx) {
  Packet p;

  for (size_t iF = 0; iF < packetType.size(); ++iF) {
    size_t field = idx % packetType[iF];
    p.push_back(field);
    idx -= field;
    idx /= packetType[iF];
  }
  return p;
}

size_t PacketSet::index(const std::set<Packet> &packets) {
  size_t idx = 0;
  for (const Packet &p : packets) {
    idx |= 1 << packetIndex(p);
  }
  return idx;
}

Eigen::VectorXd PacketSet::toVec(const std::set<Packet> &packets) {
  size_t pIdx = index(packets);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(matrixDim);
  v(pIdx) = 1;
  return v;
}

std::set<Packet> PacketSet::packetSetFromIndex(size_t idx) {
  std::set<Packet> packets;

  for (size_t iP = 0; iP < possiblePackets; ++iP) {
    if ((idx & 1) == 1) {
      packets.insert(packetFromIndex(iP));
    }
    idx = idx >> 1;
  }

  return packets;
}

TransitionMatrix PacketSet::drop() {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < matrixDim; ++i) {
    T.emplace_back(0, i, 1);
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

TransitionMatrix PacketSet::skip() { return speye(matrixDim); }

// B[fieldIndex = fieldValue]
TransitionMatrix PacketSet::test(size_t fieldIndex, size_t fieldValue) {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < matrixDim; ++i) {
    std::set<Packet> packetsIn = packetSetFromIndex(i);
    std::set<Packet> packetsOut;
    for (const Packet &p : packetsIn) {
      if (p[fieldIndex] == fieldValue) {
        packetsOut.insert(p);
      }
    }
    T.emplace_back(index(packetsOut), i, 1);
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

// B[#fieldIndex = fieldValue : n]
TransitionMatrix PacketSet::testSize(size_t fieldIndex, size_t fieldValue,
                                     size_t n) {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < matrixDim; ++i) {
    std::set<Packet> packetsIn = packetSetFromIndex(i);
    size_t nPacketsOut = 0;
    for (const Packet &p : packetsIn) {
      if (p[fieldIndex] == fieldValue) {
        nPacketsOut += 1;
      }
    }

    if (nPacketsOut == n) {
      T.emplace_back(i, i, 1);
    } else {
      T.emplace_back(i, 0, 1);
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

// B[fieldIndex <- fieldValue]
TransitionMatrix PacketSet::set(size_t fieldIndex, size_t fieldValue) {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < matrixDim; ++i) {
    std::set<Packet> packetsIn = packetSetFromIndex(i);
    std::set<Packet> packetsOut;
    for (Packet p : packetsIn) {
      p[fieldIndex] = fieldValue;
      packetsOut.insert(p);
    }
    T.emplace_back(index(packetsOut), i, 1);
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

// B[p & q]
TransitionMatrix PacketSet::amp(TransitionMatrix p, TransitionMatrix q) {

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
  std::vector<Eigen::Triplet<double>> T;

  size_t bP, bQ;
  double valP, valQ;
  for (size_t a = 0; a < matrixDim; ++a) {
    for (std::pair<size_t, double> pPair : pTriplets[a]) {
      std::tie(bP, valP) = pPair;
      for (std::pair<size_t, double> qPair : qTriplets[a]) {
        std::tie(bQ, valQ) = qPair;
        size_t packetUnion = bP | bQ;
        T.emplace_back(packetUnion, a, valP * valQ);
      }
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

// B[p;q]
TransitionMatrix PacketSet::seq(TransitionMatrix p, TransitionMatrix q) {
  return p * q;
}

// B[p \oplus_r q]
TransitionMatrix PacketSet::choice(double r, TransitionMatrix p,
                                   TransitionMatrix q) {
  return r * p + (1 - r) * q;
}

// TODO: clean up, write more tests
// TODO: normalize columns after pruning small values?
// B[p*]
TransitionMatrix PacketSet::star(TransitionMatrix p) {

  Eigen::SparseMatrix<double> S(matrixDim * matrixDim, matrixDim * matrixDim);
  Eigen::SparseMatrix<double> U(matrixDim * matrixDim, matrixDim * matrixDim);

  std::vector<Eigen::Triplet<double>> Ts, Tu;

  Vector<bool> isSaturated(matrixDim * matrixDim);
  Vector<bool> isAbsorbing(matrixDim * matrixDim);

  auto bigIndex = [&](size_t i, size_t j) { return i + matrixDim * j; };
  auto bigUnindex = [&](size_t i) {
    return std::make_pair(i % matrixDim, floor(i / matrixDim));
  };

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

      // initialize isSaturated to true. We'll mark the non-saturated things as
      // false later
      isSaturated(col) = true;
      isAbsorbing(col) = false;

      Eigen::VectorXd aVec = Eigen::VectorXd::Zero(matrixDim);
      aVec(a) = 1;
      Eigen::VectorXd aPrimeVec = p * aVec;

      for (size_t aPrime = 0; aPrime < matrixDim; ++aPrime) {
        if (abs(aPrimeVec(aPrime)) < 1e-8) {
          continue;
        }
        // Take union of sets
        size_t bPrime = a | b;
        if (bPrime != b) {
          isSaturated(col) = false;
        }
        size_t row = bigIndex(aPrime, bPrime);
        Ts.emplace_back(row, col, aPrimeVec(aPrime));
        ancestors[row].push_back(col);
      }
    }
  }

  std::function<void(size_t)> markUnsaturated = [ancestors, &isSaturated,
                                                 &markUnsaturated](size_t ind) {
    isSaturated(ind) = false;
    for (size_t a : ancestors[ind]) {
      if (isSaturated(a)) {
        markUnsaturated(a);
      }
    }
  };

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
        isAbsorbing(row) = true;
      } else {
        size_t row = bigIndex(a, b);
        Tu.emplace_back(row, col, 1);
      }
    }
  }

  S.setFromTriplets(Ts.begin(), Ts.end());
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
  // cout << Rt << endl;

  Eigen::SparseMatrix<double> limit =
      reassembleMatrix(decomp, speye(decomp.AA.rows()), X);

  // cout << limit << endl;

  Eigen::SparseMatrix<double> smallLimit(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (int k = 0; k < limit.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(limit, k); it; ++it) {
      if (abs(it.value()) < 1e-8) {
        continue;
      }
      size_t a, b, aPrime, bPrime;
      std::tie(a, b) = bigUnindex(it.col());
      std::tie(aPrime, bPrime) = bigUnindex(it.row());
      if (aPrime == 0 && b == 0) {
        // cout << "found one!  ";
        T.emplace_back(bPrime, a, it.value());
      } else {
        // cout << "bad one :'(";
      }
      // cout << "\t a: " << a << "\tb: " << b << "\ta': " << aPrime
      //      << "\tb': " << bPrime << "\tsaturated: " << isSaturated(it.col())
      //      << "\tval: " << it.value() << endl;
    }
  }

  smallLimit.setFromTriplets(T.begin(), T.end());
  return smallLimit;
}

// B[p*]
// p* is &_{i=0}^\infty p^i. We can approximate this by repeated squaring.
// Let s_k = &_{i=1}^{2^k} p^i.
// Then s_{k+1} = &_{i=1}^{2^{k+1}} p^i = s_k & (p^{2^k} * s_k)
// Then we return s_N & 1 for some big N
TransitionMatrix PacketSet::starApprox(TransitionMatrix p, double tol) {
  TransitionMatrix s = p;
  TransitionMatrix pPow = p;

  for (size_t i = 0; i < 1 / tol; ++i) {
    s = amp(s, seq(pPow, s));
    pPow = seq(pPow, pPow);
  }

  return amp(s, skip());
}
