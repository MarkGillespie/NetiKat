#include "linear_algebra_utilities.h"

SparseMatrix<double> speye(size_t n) {

  Eigen::SparseMatrix<double> M(n, n);
  std::vector<Eigen::Triplet<double>> T;

  for (size_t i = 0; i < n; ++i) {
    T.emplace_back(i, i, 1);
  }

  M.setFromTriplets(T.begin(), T.end());
  return M;
}
