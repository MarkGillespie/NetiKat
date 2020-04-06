#include "utils.h"

// Compute n choose k
// Uses recursive algorithm from here:
// https://math.stackexchange.com/questions/202554/how-do-i-compute-binomial-coefficients-efficiently
size_t binom(size_t n, size_t k) {
  if (k == 0 || k == n) {
    return 1;
  } else if (k > n) {
    throw std::invalid_argument(
        "In binom(n, k), k must be less than or equal to n. But k is " +
        std::to_string(k) + " and n is " + std::to_string(n));
  } else if (k > n / 2) {
    return binom(n, n - k);
  } else {
    return n * binom(n - 1, k - 1) / k;
  }
}

// Count the ones in a binary number n
// Uses std::bitset, described here:
// https://stackoverflow.com/questions/14682641/count-number-of-1s-in-binary-format-of-decimal-number
size_t countOnes(size_t n) {
  std::bitset<sizeof(size_t) * CHAR_BIT> b(n);
  return b.count();
}
