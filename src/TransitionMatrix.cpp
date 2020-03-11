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

TransitionMatrix PacketSet::skip() {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < matrixDim; ++i) {
    T.emplace_back(i, i, 1);
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

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

// B[p*]
TransitionMatrix PacketSet::star(TransitionMatrix p) {
  // We want to find M such that M = (p ; M) & skip
  // I claim that (p ; M) & skip is a linear function of M (if we unwrap M into
  // an n^2 x 1 column vector). Denote this by (p ; M) & skip = A M^. Then
  // we just want to solve a linear system (A-I)M^ = 0
  // We want a nonzero solution, so this becomes an eigenvector problem of A
  // with eigenvalue 1

  // Note that B[q & skip]_{a,b} = \sum_c [c \cup a = b] B[q]_{a, c}
  Eigen::SparseMatrix<double> timeP(matrixDim * matrixDim,
                                    matrixDim * matrixDim);
  Eigen::SparseMatrix<double> andSkip(matrixDim * matrixDim,
                                      matrixDim * matrixDim);
  auto unIndex = [&](size_t i) {
    return std::make_pair(i % matrixDim, floor(i / matrixDim));
  };

  for (size_t a = 0; a < matrixDim; ++a) {
    for (size_t b = 0; b < matrixDim; ++b) {
    }
  }

  return p;
}

BigTransitionMatrix PacketSet::lift(TransitionMatrix p) {
  Eigen::SparseMatrix<double> M(matrixDim * matrixDim, matrixDim * matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  // (p ; M)_{ij} = \sum_k p_{ik} M_{kj}
  for (size_t i = 0; i < matrixDim; ++i) {
    for (size_t j = 0; j < matrixDim; ++j) {
      for (size_t k = 0; k < matrixDim; ++k) {
        size_t row = bigIndex(i, j);
        size_t col = bigIndex(k, j);
        T.emplace_back(row, col, p(i, k));
      }
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

Eigen::VectorXd PacketSet::flatten(TransitionMatrix p) {
  Eigen::VectorXd v = Eigen::VectorXd::Zero(matrixDim * matrixDim);
  for (int k = 0; k < p.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(p, k); it; ++it) {
      v(bigIndex(it.row(), it.col())) = it.value();
    }
  }
  return v;
}

TransitionMatrix PacketSet::unflatten(Eigen::VectorXd v) {
  Eigen::SparseMatrix<double> M(matrixDim, matrixDim);
  std::vector<Eigen::Triplet<double>> T;

  size_t row, col;
  for (size_t iV = 0; iV < v.size(); ++iV) {
    if (abs(v(iV)) > 1e-8) {
      std::tie(row, col) = bigUnIndex(iV);
      T.emplace_back(row, col, v(iV));
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

size_t PacketSet::bigIndex(size_t i, size_t j) { return i + j * matrixDim; }
std::pair<size_t, size_t> PacketSet::bigUnIndex(size_t i) {
  return std::make_pair(i % matrixDim, floor(i / matrixDim));
}
