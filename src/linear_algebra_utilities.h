#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

#include <iostream>

// Code from geometry-central
// Nicer name for dynamic matrix
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Nicer name for sparse matrix
template <typename T> using SparseMatrix = Eigen::SparseMatrix<T>;

// Block-decompose a square sparse matrix with interleaved index sets A and B
template <typename T> struct BlockDecompositionResult {
  // The index of each element of A (resp. B) in the original system
  Vector<size_t> origIndsA;
  Vector<size_t> origIndsB;

  // Index of each orignal element in the new system  (either in the A system or
  // B)
  Vector<size_t> newInds;
  Vector<bool> isA;

  // The four "block" matrices
  SparseMatrix<T> AA;
  SparseMatrix<T> AB;
  SparseMatrix<T> BA;
  SparseMatrix<T> BB;
};
template <typename T>
BlockDecompositionResult<T> blockDecomposeSquare(const SparseMatrix<T> &m,
                                                 const Vector<bool> &Aset,
                                                 bool buildBuildBside = true);

// Apply a decomposition to a vector
template <typename T>
void decomposeVector(BlockDecompositionResult<T> &decomp, const Vector<T> &vec,
                     Vector<T> &vecAOut, Vector<T> &vecBOut);
template <typename T>
Vector<T> reassembleVector(BlockDecompositionResult<T> &decomp,
                           const Vector<T> &vecA, const Vector<T> &vecB);
template <typename T>
SparseMatrix<T> reassembleMatrix(BlockDecompositionResult<T> &decomp,
                                 const SparseMatrix<double> &AA,
                                 const SparseMatrix<double> &BA);

SparseMatrix<double> speye(size_t n);

template <typename T>
SparseMatrix<T> solveSquare(SparseMatrix<T> &A, SparseMatrix<T> &rhs);

#include "linear_algebra_utilities.ipp"
