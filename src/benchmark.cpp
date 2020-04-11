#include "benchmark.h"

int main() {

  std::vector<size_t> packetType{64};
  NetiKAT<double> neti(packetType, 4);
  benchmark(neti);

  return 0;
}
