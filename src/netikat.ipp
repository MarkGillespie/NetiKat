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
  return binomialCoefficients[k + n * (maxNumPackets + 1)];
}

template <typename T> size_t NetiKAT<T>::packetIndex(const Packet &p) const {
  size_t idx = 0;

  size_t offset = 1;
  // TODO: fix in other branches?
  for (size_t iF = 0; iF < packetType.size(); ++iF) {
    idx += p[iF] * offset;
    offset *= packetType[iF];
  }

  return idx;
}

template <typename T> Packet NetiKAT<T>::packetFromIndex(size_t idx) const {
  Packet p;
  p.reserve(packetType.size());

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

// TODO: rename
template <typename T>
Distribution<T> NetiKAT<T>::toVec(const PacketSet &packets) const {
  size_t pIdx = index(packets);
  Distribution<T> v;
  v[pIdx] = 1;
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
  TransitionMatrix<T> f = [&](Distribution<T> p) -> Distribution<T> {
    Distribution<T> pOut;
    pOut[0] = 1;
    return pOut;
  };
  return f;
}

template <typename T> TransitionMatrix<T> NetiKAT<T>::skip() const {
  TransitionMatrix<T> f = [&](Distribution<T> p) -> Distribution<T> {
    return p;
  };
  return f;
}

// B[fieldIndex = fieldValue]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::test(size_t fieldIndex,
                                     size_t fieldValue) const {
  // Capture fieldIndex and fieldValue by value, all other NetiKAT member
  // variables by reference
  TransitionMatrix<T> f = [&, fieldIndex,
                           fieldValue](Distribution<T> p) -> Distribution<T> {
    Distribution<T> pOut;
    for (std::pair<size_t, double> pair : p) {
      PacketSet packetsIn = packetSetFromIndex(pair.first);
      PacketSet packetsOut;
      for (size_t iP : packetsIn) {
        Packet pac = packetFromIndex(iP);
        if (pac[fieldIndex] == fieldValue) {
          packetsOut.insert(iP);
        }
      }
      pOut[index(packetsOut)] += pair.second;
    }
    return pOut;
  };
  return f;
}

// B[#fieldIndex = fieldValue : n]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::testSize(size_t fieldIndex, size_t fieldValue,
                                         size_t n) const {
  // Capture fieldIndex, fieldValue, and n by value, all other NetiKAT member
  // variables by reference
  TransitionMatrix<T> f = [&, fieldIndex, fieldValue,
                           n](Distribution<T> p) -> Distribution<T> {
    Distribution<T> pOut;
    for (std::pair<size_t, double> pair : p) {
      PacketSet packetsIn = packetSetFromIndex(pair.first);
      size_t nPacketsOut = 0;
      for (size_t iP : packetsIn) {
        Packet pac = packetFromIndex(iP);
        if (pac[fieldIndex] == fieldValue) {
          nPacketsOut += 1;
        }
      }
      if (nPacketsOut == n) {
        pOut[pair.first] = pair.second;
      } else {
        pOut[0] += pair.second;
      }
    }
    return pOut;
  };
  return f;
}

// B[fieldIndex <- fieldValue]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::set(size_t fieldIndex,
                                    size_t fieldValue) const {
  // Capture fieldIndex and fieldValue by value, all other NetiKAT member
  // variables by reference
  TransitionMatrix<T> f = [&, fieldIndex,
                           fieldValue](Distribution<T> p) -> Distribution<T> {
    Distribution<T> pOut;
    for (std::pair<size_t, double> pair : p) {
      PacketSet packetsIn = packetSetFromIndex(pair.first);
      size_t nPacketsOut = 0;
      PacketSet packetsOut;
      for (size_t iP : packetsIn) {
        Packet pac = packetFromIndex(iP);
        pac[fieldIndex] = fieldValue;
        packetsOut.insert(packetIndex(pac));
      }
      pOut[index(packetsOut)] += pair.second;
    }
    return pOut;
  };
  return f;
}

template <typename T> TransitionMatrix<T> NetiKAT<T>::cap1() const {
  TransitionMatrix<T> f = [&](Distribution<T> p) -> Distribution<T> {
    Distribution<T> pOut;
    for (std::pair<size_t, double> pair : p) {
      PacketSet packetsIn = packetSetFromIndex(pair.first);

      // If the set already has size 1, don't do anything
      if (packetsIn.size() <= 1) {
        pOut[pair.first] += pair.second;
      } else {
        // Otherwise, yield a set containing one of the elements with equal
        // probability
        size_t setSize = packetsIn.size();

        // Uniform distribution over packets in the set
        double prob = 1 / (double)setSize;

        for (size_t pac : packetsIn) {
          PacketSet packetsOut{pac};
          pOut[index(packetsOut)] += pair.second * prob;
        }
      }
    }
    return pOut;
  };
  return f;
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
Distribution<T> NetiKAT<T>::amp(const Distribution<T> &p,
                                const Distribution<T> &q) const {
  Distribution<T> pOut;

  for (std::pair<size_t, double> iP : p) {
    for (std::pair<size_t, double> iQ : q) {
      double prob = iP.second * iQ.second;
      if (prob > 1e-12) {
        size_t packetUnion = packetSetUnion(iP.first, iQ.first);
        pOut[packetUnion] += prob;
      }
    }
  }
  normalize(pOut);
  return pOut;
}

template <typename T>
TransitionMatrix<T> NetiKAT<T>::amp(TransitionMatrix<T> a,
                                    TransitionMatrix<T> b) const {
  // TODO: capture by reference instead?
  // Capture a and b by value so they don't disappear
  TransitionMatrix<T> f = [&, a, b](Distribution<T> p) -> Distribution<T> {
    return amp(a(p), b(p));
  };
  return f;
}

template <typename T>
TransitionMatrix<T> NetiKAT<T>::amp(std::vector<TransitionMatrix<T>> as) const {
  TransitionMatrix<T> f = [&, as](Distribution<T> p) -> Distribution<T> {
    if (as.size() == 0) {
      return p;
    } else {
      Distribution<T> pOut = as[0](p);
      for (size_t iA = 1; iA < as.size(); ++iA) {
        pOut = amp(pOut, as[iA](p));
      }
      return pOut;
    }
  };
  return f;
}

// B[p;q]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::seq(TransitionMatrix<T> a,
                                    TransitionMatrix<T> b) const {
  // TODO: capture by reference instead?
  // Capture a and b by value so they don't disappear
  TransitionMatrix<T> f = [&, a, b](Distribution<T> p) -> Distribution<T> {
    return a(b(p));
  };
  return f;
}

// B[p \oplus_r q]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::choice(T r, TransitionMatrix<T> a,
                                       TransitionMatrix<T> b) const {
  // TODO: capture by reference instead?
  // Capture a and b by value so they don't disappear
  TransitionMatrix<T> f = [&, a, b, r](Distribution<T> p) -> Distribution<T> {
    Distribution<T> ap = a(p);
    Distribution<T> bp = b(p);
    Distribution<T> pOut;
    for (std::pair<size_t, double> iP : ap) {
      pOut[iP.first] += r * iP.second;
    }
    for (std::pair<size_t, double> iP : bp) {
      pOut[iP.first] += (1 - r) * iP.second;
    }
    return pOut;
  };
  return f;
}

template <typename T>
TransitionMatrix<T> NetiKAT<T>::ifThen(TransitionMatrix<T> test,
                                       TransitionMatrix<T> body) const {
  return seq(body, test);
}

// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::starApprox(TransitionMatrix<T> a, T tol) const {

  auto dist = [](Distribution<T> a, const Distribution<T> &b) {
    for (std::pair<size_t, double> iB : b) {
      a[iB.first] -= iB.second;
    }
    double sumSq = 0;
    for (std::pair<size_t, double> iA : a) {
      sumSq += pow(iA.second, 2);
    }
    return sqrt(sumSq);
  };

  // TODO: capture by reference instead?
  // Capture a and b by value so they don't disappear
  TransitionMatrix<T> f = [&, a, tol](Distribution<T> p) -> Distribution<T> {
    Distribution<T> oldS = p;
    Distribution<T> s = amp(p, a(oldS));
    starIter = 0;
    while (dist(s, oldS) > tol) {
      oldS = s;
      s = amp(p, a(s));
      normalize(s);
      starIter++;
    }
    return s;
  };
  return f;
}

// B[p*]
template <typename T>
TransitionMatrix<T> NetiKAT<T>::dumbStarApprox(TransitionMatrix<T> a,
                                               size_t iter) const {
  // TODO: capture by reference instead?
  // Capture a and b by value so they don't disappear
  TransitionMatrix<T> f = [&, a, iter](Distribution<T> p) -> Distribution<T> {
    Distribution<T> s = p;
    for (size_t i = 0; i < iter; ++i) {
      s = amp(p, a(s));
      normalize(s);
    }
    return s;
  };
  return f;
}

// Make p a probability distribution by normalizing its column sum to 1
template <typename T> void NetiKAT<T>::normalize(Distribution<T> &p) const {

  double sum = 0;
  // Iterate over an unordered_map using range based for loop
  for (std::pair<std::size_t, T> entry : p) {
    p[entry.first] = abs(entry.second); // TODO: is this necessary? is it slow?
    sum += p[entry.first];
  }
  for (std::pair<std::size_t, T> entry : p) {
    p[entry.first] = p[entry.first] / sum;
  }
}
