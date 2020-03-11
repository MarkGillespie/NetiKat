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

#endif
