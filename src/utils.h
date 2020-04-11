#pragma once

#include <bitset> // count ones in a bitset
#include <iostream>
#include <stddef.h>  // size_t
#include <stdexcept> // invalid_argument
#include <stdlib.h>  // rand
#include <string>    // string
#include <unordered_map>

#include <Eigen/Dense>

// Compute n choose k
size_t binom(size_t n, size_t k);

// Count the ones in the binary number n
size_t countOnes(size_t n);

// Generate a random floating point number between fMin and fMax
template <typename T> T fRand(T fMin, T fMax);

template <typename T>
std::unordered_map<size_t, T> randMap(size_t n, size_t nEntries, T fMin,
                                      T fMax);

// Return a map whose entries sum to 1
template <typename T>
std::unordered_map<size_t, T> randNormalizedMap(size_t n, size_t nEntries);

// Return a map whose entries sum to 1
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
toVec(const std::unordered_map<size_t, T> &m);

#include "utils.ipp"
