#pragma once

#include <iomanip> // std::setprecision
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <bitset>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "linear_algebra_utilities.h"
#include "utils.h"

using std::cerr;
using std::cout;
using std::endl;

struct lex_compare {};

using TransitionMatrix = Eigen::SparseMatrix<double>;

// Number of possible values for each field
using PacketType = std::vector<size_t>;

// Particular entry in each field
using Packet = std::vector<size_t>;

// The indices of the packets contained in this set
using PacketSet = std::set<size_t>;

class NetiKAT {
public:
  NetiKAT(const PacketType &type_, size_t maxNumPackets_);

  PacketType packetType;
  size_t matrixDim;
  size_t possiblePackets;
  size_t maxNumPackets;

  // numNetiKATsOfSizeLessThan[i] is the number of packet sets of size less
  // than i
  // This is useful when indexing packet sets
  std::vector<size_t> numNetiKATsOfSizeLessThan;

  std::vector<size_t> binomialCoefficients;
  size_t binomialCoefficient(size_t n, size_t k) const;

  size_t packetIndex(const Packet &p) const;
  Packet packetFromIndex(size_t idx) const;

  // The indexing scheme for packet sets is as follows:
  // First of all, packet sets are ordered by cardinality. Within each
  // collection of packet sets with the same cardinality, the packet sets are
  // ordered lexicographically
  // Packet sets which are too big get index 0
  size_t index(const PacketSet &packets) const;
  Eigen::VectorXd toVec(const PacketSet &packets) const;
  PacketSet packetSetFromIndex(size_t idx) const;

  size_t bigIndex(size_t i, size_t j) const;
  std::pair<size_t, size_t> bigUnindex(size_t i) const;

  // Compute the union of the packet set with index a and the packet set with
  // index b
  size_t packetSetUnion(size_t a, size_t b) const;

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
