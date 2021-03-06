#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <gtest/gtest.h>

#include <Eigen/SparseCore>

#include <stdlib.h> /* rand */

#define EXPECT_MAT_EQ(a, b) ExpectMatEq(a, b)
#define EXPECT_MAT_NEAR(a, b, eps) ExpectMatEq(a, b, eps)

using std::cerr;
using std::cout;
using std::endl;
using std::string;

double fRand(double fMin, double fMax) {
  double f = (double)rand() / (RAND_MAX + 1.0);
  return fMin + f * (fMax - fMin);
}

Eigen::SparseMatrix<double> randomDenseStochastic(size_t n) {
  Eigen::SparseMatrix<double> M(n, n);
  std::vector<Eigen::Triplet<double>> T;

  T.emplace_back(0, 0, 1);
  for (size_t col = 1; col < n; ++col) {
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
      v(i) = abs(v(i));
      sum += v(i);
    }
    v /= sum;
    for (size_t row = 0; row < n; ++row) {
      T.emplace_back(row, col, v(row));
    }
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}

bool isStochastic(Eigen::SparseMatrix<double> M) {
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(M.rows());

  return (M.transpose() * ones - ones).norm() < 1e-8;
}

// =============================================================
//                Template Magic for Eigen
// =============================================================
// googletest doesn't like printing out eigen matrices when they fail tests
// so I stole this code from
// https://stackoverflow.com/questions/25146997/teach-google-test-how-to-print-eigen-matrix

template <class Base> class EigenPrintWrap : public Base {
  friend void PrintTo(const EigenPrintWrap &m, ::std::ostream *o) {
    size_t width = (m.cols() < 10) ? m.cols() : 10;
    size_t height = (m.rows() < 10) ? m.rows() : 10;
    *o << "\n" << m.topLeftCorner(height, width) << "...";
  }
};

template <class Base> const EigenPrintWrap<Base> &print_wrap(const Base &base) {
  return static_cast<const EigenPrintWrap<Base> &>(base);
}

bool MatrixEq(const EigenPrintWrap<Eigen::MatrixXd> &lhs_,
              const EigenPrintWrap<Eigen::MatrixXd> &rhs_, double difference,
              double threshold = -1) {
  Eigen::MatrixXd lhs = static_cast<Eigen::MatrixXd>(lhs_);
  Eigen::MatrixXd rhs = static_cast<Eigen::MatrixXd>(rhs_);
  double err = (lhs - rhs).norm();
  if (threshold > 0) {
    bool equal = abs(err) < threshold;
    if (!equal)
      cerr << "norm of difference: " << err << endl;
    return equal;
  } else {
    const ::testing::internal::FloatingPoint<double> difference(err), zero(0);
    bool equal = difference.AlmostEquals(zero);
    if (!equal)
      cerr << "norm of difference: " << err << endl;
    return equal;
  }
}

void ExpectMatEq(const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_,
                 double threshold = -1) {
  EXPECT_PRED4(MatrixEq, print_wrap(a_), print_wrap(b_), (a_ - b_).norm(),
               threshold);
}

Eigen::SparseMatrix<double> randomPositiveSparse(size_t n, double density) {
  size_t entries = (size_t)(density * pow((double)n, 2));
  std::vector<Eigen::Triplet<double>> T;

  for (size_t iE = 0; iE < entries; ++iE) {
    size_t row = rand() % n;
    size_t col = rand() % n;
    double entry = fRand(0, 5);

    T.emplace_back(col, row, entry);
  }
  Eigen::SparseMatrix<double> M(n, n);
  M.setFromTriplets(T.begin(), T.end());
  return M;
}

#endif
