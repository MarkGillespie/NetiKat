#include "TransitionMatrix.h"
#include "utils.h"

#include <bitset>

class TransitionMatrixTest : public testing::Test {
public:
  static std::unique_ptr<PacketSet> set;

protected:
  static void SetUpTestSuite() {
    std::vector<size_t> packetType{2, 2};
    set = std::unique_ptr<PacketSet>(new PacketSet(packetType, 4));
  }

  void SetUp() override {}

  static void TearDownTestSuite() { set.release(); }
};

std::unique_ptr<PacketSet> TransitionMatrixTest::set = nullptr;

// ============================================
//        TransitionMatrix Tests
// ============================================
TEST_F(TransitionMatrixTest, binomialCoefficients) {
  size_t nCk = binom(56, 12);
  size_t answer = 558383307300; // Solution computed in Mathematica

  EXPECT_EQ(nCk, answer);
}

TEST_F(TransitionMatrixTest, normalizedMatrixIsStochastic) {
  size_t dim = 10;
  TransitionMatrix A = randomPositiveSparse(dim, 0.5);

  set->normalize(A);
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(dim);
  Eigen::VectorXd colSums = A.transpose() * ones;

  EXPECT_MAT_NEAR(colSums, ones, 1e-8);
}

TEST_F(TransitionMatrixTest, indexOfPacketFromIndexIsIdentity) {
  for (size_t iP = 0; iP < set->possiblePackets; ++iP) {
    size_t newIndex = set->packetIndex(set->packetFromIndex(iP));
    EXPECT_EQ(iP, newIndex);
  }
}

TEST_F(TransitionMatrixTest, compressDecompressIndexIsIdentity) {
  std::vector<size_t> biggerPacketType{4, 4};
  PacketSet bigSet(biggerPacketType, 3);

  for (size_t iS = 0; iS < bigSet.matrixDim; iS++) {
    size_t decompressedIndex = bigSet.decompressIndex(iS);
    size_t nOnes = countOnes(decompressedIndex);
    size_t newIdx = bigSet.compressIndex(decompressedIndex, nOnes);

    ASSERT_EQ(iS, newIdx);
  }
}

TEST_F(TransitionMatrixTest, decompressCompressIndexIsIdentity) {
  std::vector<size_t> biggerPacketType{4, 4, 4, 4};
  PacketSet bigSet(biggerPacketType, 4);

  size_t idx = 6;
  size_t compressedIndex = bigSet.compressIndex(idx, 2);
  size_t newIdx = bigSet.decompressIndex(compressedIndex);

  EXPECT_EQ(idx, newIdx);
}

TEST_F(TransitionMatrixTest, zeroCompressesToZero) {
  size_t zeroIdx = set->compressIndex(0, 0);

  EXPECT_EQ(zeroIdx, 0);
}

TEST_F(TransitionMatrixTest, indexOfPacketSetFromIndexIsIdentity) {
  for (size_t iP = 0; iP < set->matrixDim; ++iP) {
    size_t newIndex = set->index(set->packetSetFromIndex(iP));
    EXPECT_EQ(iP, newIndex);
  }
}

TEST_F(TransitionMatrixTest, bigIndexUnindexInverse) {
  size_t i, j;
  for (size_t iP = 0; iP < set->matrixDim * set->matrixDim; ++iP) {
    std::tie(i, j) = set->bigUnindex(iP);
    size_t newIndex = set->bigIndex(i, j);
    EXPECT_EQ(iP, newIndex);
  }
}

TEST_F(TransitionMatrixTest, Skip) {
  TransitionMatrix skipMat = set->skip();

  // Generate a random probability distribution
  Eigen::VectorXd v = Eigen::VectorXd::Random(set->matrixDim);
  for (size_t i = 0; i < set->matrixDim; ++i) {
    v(i) = abs(v(i));
  }
  v /= v.lpNorm<1>();

  Eigen::VectorXd skipV = skipMat * v;

  EXPECT_MAT_EQ(v, skipV);
}

TEST_F(TransitionMatrixTest, Drop) {
  TransitionMatrix dropMat = set->drop();
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1}});
  Eigen::VectorXd droppedVec = dropMat * v;

  Eigen::VectorXd trueDroppedVec = set->toVec(std::set<Packet>());

  EXPECT_MAT_EQ(droppedVec, trueDroppedVec);
}

TEST_F(TransitionMatrixTest, Test) {
  TransitionMatrix testMat = set->test(0, 1);
  Eigen::VectorXd v = set->toVec(
      std::set<Packet>{std::vector<size_t>{0}, std::vector<size_t>{1}});
  Eigen::VectorXd testedVec = testMat * v;

  Eigen::VectorXd trueTestedVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{1}});

  EXPECT_MAT_EQ(testedVec, trueTestedVec);
}

TEST_F(TransitionMatrixTest, TestSize) {
  TransitionMatrix testMat = set->testSize(0, 1, 2);
  Eigen::VectorXd v = set->toVec(
      std::set<Packet>{std::vector<size_t>{1, 1}, std::vector<size_t>{1}});
  Eigen::VectorXd testedVec = testMat * v;

  EXPECT_MAT_EQ(testedVec, v);
}

TEST_F(TransitionMatrixTest, Set) {
  TransitionMatrix setMat = set->set(0, 1);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{0}});
  Eigen::VectorXd setVec = setMat * v;

  Eigen::VectorXd trueSetVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{1}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, Amp) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  TransitionMatrix setBothMat = set->amp(setZeroMat, setOneMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{0}});
  Eigen::VectorXd setVec = setBothMat * v;

  Eigen::VectorXd trueSetVec = set->toVec(
      std::set<Packet>{std::vector<size_t>{0}, std::vector<size_t>{1}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, AmpPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  TransitionMatrix B = randomDenseStochastic(set->matrixDim);
  EXPECT_TRUE(isStochastic(A));
  EXPECT_TRUE(isStochastic(B));

  TransitionMatrix C = set->amp(A, B);
  EXPECT_TRUE(isStochastic(C));
}

TEST_F(TransitionMatrixTest, AmpCommutes) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  TransitionMatrix B = randomDenseStochastic(set->matrixDim);

  TransitionMatrix C = set->amp(A, B);
  TransitionMatrix D = set->amp(B, A);
  EXPECT_MAT_NEAR(C, D, 1e-12);
}

TEST_F(TransitionMatrixTest, Seq) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  TransitionMatrix setZeroThenOneMat = set->seq(setOneMat, setZeroMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1}});

  Eigen::VectorXd setZeroVec = setZeroMat * v;
  Eigen::VectorXd trueSetZeroVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{0}});
  EXPECT_MAT_EQ(setZeroVec, trueSetZeroVec);

  Eigen::VectorXd setVec = setZeroThenOneMat * v;
  Eigen::VectorXd trueSetVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{1}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, SeqPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  TransitionMatrix B = randomDenseStochastic(set->matrixDim);
  EXPECT_TRUE(isStochastic(A));
  EXPECT_TRUE(isStochastic(B));

  TransitionMatrix C = set->seq(A, B);
  EXPECT_TRUE(isStochastic(C));
}

TEST_F(TransitionMatrixTest, Choice) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  double p = 0.25;
  TransitionMatrix probMat = set->choice(p, setZeroMat, setOneMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1}});
  Eigen::VectorXd v0 = set->toVec(std::set<Packet>{std::vector<size_t>{0}});
  Eigen::VectorXd v1 = set->toVec(std::set<Packet>{std::vector<size_t>{1}});

  Eigen::VectorXd chosenVec = probMat * v;
  Eigen::VectorXd trueChosenVec = p * v0 + (1 - p) * v1;

  EXPECT_MAT_EQ(chosenVec, trueChosenVec);
}

TEST_F(TransitionMatrixTest, StarPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  EXPECT_TRUE(isStochastic(A));

  TransitionMatrix B = set->star(A);
  EXPECT_TRUE(isStochastic(B));
}

TEST_F(TransitionMatrixTest, StarAgreesWithStarApprox) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  TransitionMatrix star = set->star(A);
  TransitionMatrix approxStar = set->starApprox(A, 1e-12);

  EXPECT_MAT_NEAR(star, approxStar, 1e-12);
}

TEST_F(TransitionMatrixTest, StarApproxCountsIterationsCorrectly) {
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  size_t iter;
  TransitionMatrix approxStar = set->starApprox(A, 1e-5, iter);
  TransitionMatrix iterStar = set->dumbStarApprox(A, iter);

  EXPECT_MAT_NEAR(iterStar, approxStar, 1e-12);
}

TEST_F(TransitionMatrixTest, StarApproxKindaConverged) {
  double tol = 1e-12;
  TransitionMatrix A = randomDenseStochastic(set->matrixDim);
  size_t iter;
  TransitionMatrix approxStar = set->starApprox(A, tol, iter);
  TransitionMatrix nextApproxStar = set->dumbStarApprox(A, iter + 1);

  EXPECT_MAT_NEAR(nextApproxStar, approxStar, tol);
}
