#include <bitset> // count ones in a bitset
#include <iostream>
#include <stddef.h>  // size_t
#include <stdexcept> // invalid_argument
#include <string>    // string

// Compute n choose k
size_t binom(size_t n, size_t k);

// Count the ones in the binary number n
size_t countOnes(size_t n);
