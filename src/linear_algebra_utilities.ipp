template <typename T>
BlockDecompositionResult<T> blockDecomposeSquare(const SparseMatrix<T> &m,
                                                 const Vector<bool> &Aset,
                                                 bool buildBuildBside) {

  if (m.rows() != m.cols())
    throw std::logic_error(
        "blockDecomposeSquare must be called on square matrix");

  // Count sizes
  size_t initSize = m.rows();
  size_t Asize = 0;
  size_t Bsize = 0;
  for (size_t i = 0; i < initSize; i++) {
    if (Aset[i]) {
      Asize++;
    } else {
      Bsize++;
    }
  }

  // Create the result object
  BlockDecompositionResult<T> r;
  r.isA = Aset;
  r.newInds = Vector<size_t>(initSize);
  r.origIndsA = Vector<size_t>(Asize);
  r.origIndsB = Vector<size_t>(Bsize);
  r.AA = SparseMatrix<T>(Asize, Asize);
  r.AB = SparseMatrix<T>(Asize, Bsize);
  if (buildBuildBside) {
    r.BA = SparseMatrix<T>(Bsize, Asize);
    r.BB = SparseMatrix<T>(Bsize, Bsize);
  } else {
    r.BA = SparseMatrix<T>(0, 0);
    r.BB = SparseMatrix<T>(0, 0);
  }

  // Index
  size_t Aind = 0;
  size_t Bind = 0;
  for (size_t i = 0; i < initSize; i++) {
    if (Aset[i]) {
      r.origIndsA[Aind] = i;
      r.newInds[i] = Aind;
      Aind++;
    } else {
      r.origIndsB[Bind] = i;
      r.newInds[i] = Bind;
      Bind++;
    }
  }

  // Split
  std::vector<Eigen::Triplet<T>> AAtrip;
  std::vector<Eigen::Triplet<T>> ABtrip;
  std::vector<Eigen::Triplet<T>> BAtrip;
  std::vector<Eigen::Triplet<T>> BBtrip;
  for (size_t k = 0; k < (size_t)m.outerSize(); k++) {
    for (typename SparseMatrix<T>::InnerIterator it(m, k); it; ++it) {

      size_t rowInd = it.row();
      size_t colInd = it.col();
      T val = it.value();

      bool rowA = Aset[rowInd];
      bool colA = Aset[colInd];

      if (rowA && colA) {
        AAtrip.emplace_back(r.newInds[rowInd], r.newInds[colInd], val);
      }
      if (rowA && !colA) {
        ABtrip.emplace_back(r.newInds[rowInd], r.newInds[colInd], val);
      }
      if (buildBuildBside) {
        if (!rowA && colA) {
          BAtrip.emplace_back(r.newInds[rowInd], r.newInds[colInd], val);
        }
        if (!rowA && !colA) {
          BBtrip.emplace_back(r.newInds[rowInd], r.newInds[colInd], val);
        }
      }
    }
  }

  // Build new matrices
  r.AA.setFromTriplets(AAtrip.begin(), AAtrip.end());
  r.AB.setFromTriplets(ABtrip.begin(), ABtrip.end());
  if (buildBuildBside) {
    r.BA.setFromTriplets(BAtrip.begin(), BAtrip.end());
    r.BB.setFromTriplets(BBtrip.begin(), BBtrip.end());
  }

  return r;
}

template <typename T>
void decomposeVector(BlockDecompositionResult<T> &decomp, const Vector<T> &vec,
                     Vector<T> &vecAOut, Vector<T> &vecBOut) {

  vecAOut = Vector<T>(decomp.origIndsA.rows());
  vecBOut = Vector<T>(decomp.origIndsB.rows());

  for (size_t i = 0; i < (size_t)vecAOut.rows(); i++) {
    vecAOut[i] = vec[decomp.origIndsA[i]];
  }
  for (size_t i = 0; i < (size_t)vecBOut.rows(); i++) {
    vecBOut[i] = vec[decomp.origIndsB[i]];
  }
}

template <typename T>
Vector<T> reassembleVector(BlockDecompositionResult<T> &decomp,
                           const Vector<T> &vecA, const Vector<T> &vecB) {

  Vector<T> vecOut(decomp.newInds.rows());

  for (size_t i = 0; i < (size_t)vecA.rows(); i++) {
    vecOut[decomp.origIndsA[i]] = vecA[i];
  }
  for (size_t i = 0; i < (size_t)vecB.rows(); i++) {
    vecOut[decomp.origIndsB[i]] = vecB[i];
  }

  return vecOut;
}

template <typename T>
SparseMatrix<T> reassembleMatrix(BlockDecompositionResult<T> &decomp,
                                 const SparseMatrix<double> &AA,
                                 const SparseMatrix<double> &AB) {

  SparseMatrix<double> M(decomp.newInds.rows(), decomp.newInds.rows());
  std::vector<Eigen::Triplet<double>> Tr;

  for (int k = 0; k < AA.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(AA, k); it; ++it) {
      size_t row = decomp.origIndsA[it.row()];
      size_t col = decomp.origIndsA[it.col()];
      Tr.emplace_back(row, col, it.value());
    }
  }

  for (int k = 0; k < AB.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(AB, k); it; ++it) {
      size_t row = decomp.origIndsA[it.row()];
      size_t col = decomp.origIndsB[it.col()];
      Tr.emplace_back(row, col, it.value());
    }
  }

  M.setFromTriplets(Tr.begin(), Tr.end());
  return M;
}

template <typename T>
SparseMatrix<T> solveSquare(SparseMatrix<T> &A, SparseMatrix<T> &rhs) {
  A.makeCompressed();
  rhs.makeCompressed();

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);

  if (solver.info() != Eigen::Success) {
    std::cerr << "Solver factorization error: " << solver.info() << std::endl;
    throw std::invalid_argument("Solver factorization failed");
  }

  return solver.solve(rhs);
}
