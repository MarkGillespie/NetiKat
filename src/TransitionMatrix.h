#pragma once

#include <iomanip> // std::setprecision
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "linear_algebra_utilities.h"

using std::cerr;
using std::cout;
using std::endl;

using TransitionMatrix = Eigen::SparseMatrix<double>;

// Number of possible values for each field
using PacketType = std::vector<size_t>;

// Particular entry in each field
using Packet = std::vector<size_t>;

class PacketSet {
public:
  PacketSet(const PacketType &type_);

  PacketType packetType;
  size_t matrixDim;
  size_t possiblePackets;

  size_t packetIndex(const Packet &p) const;
  Packet packetFromIndex(size_t idx) const;

  size_t index(const std::set<Packet> &packets) const;
  Eigen::VectorXd toVec(const std::set<Packet> &packets) const;
  std::set<Packet> packetSetFromIndex(size_t idx) const;

  size_t bigIndex(size_t i, size_t j) const;
  std::pair<size_t, size_t> bigUnindex(size_t i) const;

  TransitionMatrix drop() const;
  TransitionMatrix skip() const;

  // B[fieldIndex = fieldValue]
  TransitionMatrix test(size_t fieldIndex, size_t fieldValue) const;

  // B[#fieldIndex = fieldValue : n]
  TransitionMatrix testSize(size_t fieldIndex, size_t fieldValue,
                            size_t n) const;

  // B[fieldIndex <- fieldValue]
  TransitionMatrix set(size_t fieldIndex, size_t fieldValue) const;

  // B[p & q]
  // TODO: this is stupidly show
  TransitionMatrix amp(TransitionMatrix p, TransitionMatrix q) const;

  // B[p;q]
  TransitionMatrix seq(TransitionMatrix p, TransitionMatrix q) const;

  // B[p \oplus_r q]
  TransitionMatrix choice(double r, TransitionMatrix p,
                          TransitionMatrix q) const;

  // B[p*]
  TransitionMatrix star(TransitionMatrix p) const;

  // B[p*]
  // Computes B[p*] by computing B[p^(n)] until it differs from B[p^(n-1)] by
  // less than tol
  TransitionMatrix starApprox(TransitionMatrix p, double tol = 1e-12) const;

  // B[p*]
  // Computes B[p*] by computing B[p^(n)] until it differs from B[p^(n-1)] by
  // less than tol. Also returns how many iterations were needed
  TransitionMatrix starApprox(TransitionMatrix p, double tol,
                              size_t &iterationsNeeded) const;

  // B[p*]
  // Computes B[p^(iter)]
  TransitionMatrix dumbStarApprox(TransitionMatrix p, size_t iter) const;

  // Make M stochastic by normalizing its columns to sum to 1
  void normalize(TransitionMatrix &M) const;
};
