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
// TODO: this is stupidly show
TransitionMatrix PacketSet::amp(TransitionMatrix p, TransitionMatrix q) {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (int kP = 0; kP < p.outerSize(); ++kP) {
    for (Eigen::SparseMatrix<double>::InnerIterator itP(p, kP); itP; ++itP) {
      for (int kQ = 0; kQ < q.outerSize(); ++kQ) {
        for (Eigen::SparseMatrix<double>::InnerIterator itQ(q, kQ); itQ;
             ++itQ) {

          if (itP.col() == itQ.col()) {
            std::set<Packet> pPackets = packetSetFromIndex(itP.row());
            std::set<Packet> qPackets = packetSetFromIndex(itQ.row());

            // Take union of sets
            std::copy(qPackets.begin(), qPackets.end(),
                      std::inserter(pPackets, pPackets.end()));
            size_t d = index(pPackets);
            T.emplace_back(d, itP.col(), itP.value() * itQ.value());
          }
        }
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
    std::set<Packet> aSet = packetSetFromIndex(a);
    for (size_t b = 0; b < matrixDim; ++b) {
      size_t col = bigIndex(a, b);

      // initialize isSaturated to true. We'll mark the non-saturated things as
      // false later
      isSaturated(col) = true;
      isAbsorbing(col) = false;

      std::set<Packet> bSet = packetSetFromIndex(b);
      Eigen::VectorXd aVec = Eigen::VectorXd::Zero(matrixDim);
      aVec(a) = 1;
      Eigen::VectorXd aPrimeVec = p * aVec;

      for (size_t aPrime = 0; aPrime < matrixDim; ++aPrime) {
        if (abs(aPrimeVec(aPrime)) < 1e-8) {
          continue;
        }
        std::set<Packet> bPrimeSet(bSet.begin(), bSet.end());
        // Take union of sets
        std::copy(aSet.begin(), aSet.end(),
                  std::inserter(bPrimeSet, bPrimeSet.end()));

        size_t bPrime = index(bPrimeSet);
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
  // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
      solver;

  Rt.makeCompressed();
  IminusQ.makeCompressed();
  solver.compute(IminusQ);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Solver factorization error: " << solver.info() << std::endl;
    throw std::invalid_argument("Solver factorization failed");
  }

  Eigen::SparseMatrix<double> X = solver.solve(Rt).transpose();
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
