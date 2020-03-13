#pragma once

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

  size_t packetIndex(const Packet &p);
  Packet packetFromIndex(size_t idx);

  size_t index(const std::set<Packet> &packets);
  Eigen::VectorXd toVec(const std::set<Packet> &packets);
  std::set<Packet> packetSetFromIndex(size_t idx);

  TransitionMatrix drop();
  TransitionMatrix skip();

  // B[fieldIndex = fieldValue]
  TransitionMatrix test(size_t fieldIndex, size_t fieldValue);

  // B[#fieldIndex = fieldValue : n]
  TransitionMatrix testSize(size_t fieldIndex, size_t fieldValue, size_t n);

  // B[fieldIndex <- fieldValue]
  TransitionMatrix set(size_t fieldIndex, size_t fieldValue);

  // B[p & q]
  // TODO: this is stupidly show
  TransitionMatrix amp(TransitionMatrix p, TransitionMatrix q);

  // B[p;q]
  TransitionMatrix seq(TransitionMatrix p, TransitionMatrix q);

  // B[p \oplus_r q]
  TransitionMatrix choice(double r, TransitionMatrix p, TransitionMatrix q);

  // B[p*]
  TransitionMatrix star(TransitionMatrix p);

  // B[p*]
  TransitionMatrix starApprox(TransitionMatrix p, double tol = 1e-8);
};
