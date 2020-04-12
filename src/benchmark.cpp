#include "benchmark.h"

int main() {

  for (size_t n = 16; n <= 64; n += 4) {
    for (size_t i = 5; i < 6; ++i) {
      std::vector<size_t> packetType{n};
      NetiKAT<double> neti(packetType, i);
      benchmark(neti, 24);
    }
  }
  // std::vector<size_t> packetType{64};
  // NetiKAT<double> neti(packetType, 6);
  // benchmark(neti, 10);

  return 0;
}
