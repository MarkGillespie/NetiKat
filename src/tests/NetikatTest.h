#include "netikat.h"
#include "utils.h"

#include <bitset>

class NetikatTest : public testing::Test {
public:
  static std::unique_ptr<NetiKAT<double>> neti;

protected:
  static void SetUpTestSuite() {
    std::vector<size_t> packetType{2, 2};
    neti = std::unique_ptr<NetiKAT<double>>(new NetiKAT<double>(packetType, 2));
  }

  void SetUp() override {}

  static void TearDownTestSuite() { neti.release(); }

  static inline size_t pkt(size_t a);
  static inline size_t pkt(size_t a, size_t b);
};

std::unique_ptr<NetiKAT<double>> NetikatTest::neti = nullptr;

size_t NetikatTest::pkt(size_t a) {
  return neti->packetIndex(std::vector<size_t>{a, 0});
}
size_t NetikatTest::pkt(size_t a, size_t b) {
  return neti->packetIndex(std::vector<size_t>{a, b});
}

// ============================================
//               NetiKAT Tests
// ============================================

TEST_F(NetikatTest, binomialCoefficients) {
  size_t nCk = binom(56, 12);
  size_t answer = 558383307300; // Solution computed in Mathematica

  EXPECT_EQ(nCk, answer);
}

TEST_F(NetikatTest, cachedBinomialCoefficients) {
  NetiKAT<double> net(std::vector<size_t>{10}, 2);
  size_t nCk = net.binomialCoefficient(10, 2);
  size_t answer = 45; // 10 * 9 / 2

  EXPECT_EQ(nCk, answer);
}

TEST_F(NetikatTest, normalizedVectorIsStochastic) {
  size_t dim = neti->matrixDim;
  Distribution<double> v = randMap(dim, 24, 0.0, 1.1);

  neti->normalize(v);

  double colSum = 0;
  // Iterate over an unordered_map using range based for loop
  for (std::pair<std::size_t, double> entry : v) {
    colSum += entry.second;
  }

  EXPECT_NEAR(colSum, 1, 1e-8);
}

TEST_F(NetikatTest, indexOfPacketFromIndexIsIdentity) {
  for (size_t iP = 0; iP < neti->possiblePackets; ++iP) {
    size_t newIndex = neti->packetIndex(neti->packetFromIndex(iP));
    EXPECT_EQ(iP, newIndex);
  }
}

TEST_F(NetikatTest, indexOfPacketSetFromIndexIsIdentity) {
  for (size_t iP = 0; iP < neti->matrixDim; ++iP) {
    size_t newIndex = neti->index(neti->packetSetFromIndex(iP));
    ASSERT_EQ(iP, newIndex);
  }
}

TEST_F(NetikatTest, Skip) {
  TransitionMatrix<double> skipMat = neti->skip();

  // Generate a random probability distribution
  Distribution<double> v = randNormalizedMap<double>(neti->matrixDim, 24);

  Distribution<double> skipV = skipMat(v);

  Eigen::VectorXd vVec = toVec(v);
  Eigen::VectorXd skipVVec = toVec(skipV);
  EXPECT_MAT_EQ(vVec, skipVVec);
}

TEST_F(NetikatTest, Drop) {
  TransitionMatrix<double> dropMat = neti->drop();
  Distribution<double> v = neti->toVec(PacketSet{pkt(1)});

  Eigen::VectorXd droppedVec = toVec(dropMat(v));

  Eigen::VectorXd trueDroppedVec = toVec(neti->toVec(PacketSet()));

  EXPECT_MAT_EQ(droppedVec, trueDroppedVec);
}

TEST_F(NetikatTest, Test) {
  TransitionMatrix<double> testMat = neti->test(0, 1);
  Distribution<double> v = neti->toVec(PacketSet{pkt(0), pkt(1)});

  Eigen::VectorXd testedVec = toVec(testMat(v));
  Eigen::VectorXd trueTestedVec = toVec(neti->toVec(PacketSet{pkt(1)}));

  EXPECT_MAT_EQ(testedVec, trueTestedVec);
}

TEST_F(NetikatTest, TestSize) {
  TransitionMatrix<double> testMat = neti->testSize(0, 1, 2);
  Distribution<double> v = neti->toVec(PacketSet{pkt(1, 1), pkt(1, 0)});
  Eigen::VectorXd vVec = toVec(v);
  Eigen::VectorXd testedVec = toVec(testMat(v));

  EXPECT_MAT_EQ(testedVec, vVec);
}

TEST_F(NetikatTest, Set) {
  TransitionMatrix<double> setMat = neti->set(0, 1);
  Distribution<double> v = neti->toVec(PacketSet{pkt(0)});
  Eigen::VectorXd setVec = toVec(setMat(v));

  Eigen::VectorXd trueSetVec = toVec(neti->toVec(PacketSet{pkt(1)}));

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, Amp) {
  TransitionMatrix<double> setZeroMat = neti->set(0, 0);
  TransitionMatrix<double> setOneMat = neti->set(0, 1);
  TransitionMatrix<double> setBothMat = neti->amp(setZeroMat, setOneMat);
  Distribution<double> v = neti->toVec(PacketSet{pkt(0)});

  Eigen::VectorXd setVec = toVec(setBothMat(v));
  Eigen::VectorXd trueSetVec = toVec(neti->toVec(PacketSet{pkt(0), pkt(1)}));

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, AmpPreservesStochastic) {
  TransitionMatrix<double> A = randomDenseStochasticOperator(neti->matrixDim);
  TransitionMatrix<double> B = randomDenseStochasticOperator(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A, neti->matrixDim));
  EXPECT_TRUE(isStochastic(B, neti->matrixDim));

  TransitionMatrix<double> C = neti->amp(A, B);
  EXPECT_TRUE(isStochastic(C, neti->matrixDim));
}

TEST_F(NetikatTest, AmpCommutes) {
  TransitionMatrix<double> A = randomDenseStochasticOperator(neti->matrixDim);
  TransitionMatrix<double> B = randomDenseStochasticOperator(neti->matrixDim);

  TransitionMatrix<double> C = neti->amp(A, B);
  TransitionMatrix<double> D = neti->amp(B, A);
  EXPECT_MAT_NEAR(toMat(C, neti->matrixDim), toMat(D, neti->matrixDim), 1e-12);
}

TEST_F(NetikatTest, Seq) {
  TransitionMatrix<double> setZeroMat = neti->set(0, 0);
  TransitionMatrix<double> setOneMat = neti->set(0, 1);
  TransitionMatrix<double> setZeroThenOneMat = neti->seq(setOneMat, setZeroMat);
  Distribution<double> v = neti->toVec(PacketSet{pkt(1)});

  Eigen::VectorXd setZeroVec = toVec(setZeroMat(v));
  Eigen::VectorXd trueSetZeroVec = toVec(neti->toVec(PacketSet{pkt(0)}));
  EXPECT_MAT_EQ(setZeroVec, trueSetZeroVec);

  Eigen::VectorXd setVec = toVec(setZeroThenOneMat(v));
  Eigen::VectorXd trueSetVec = toVec(neti->toVec(PacketSet{pkt(1)}));

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, SeqPreservesStochastic) {
  TransitionMatrix<double> A = randomDenseStochasticOperator(neti->matrixDim);
  TransitionMatrix<double> B = randomDenseStochasticOperator(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A, neti->matrixDim));
  EXPECT_TRUE(isStochastic(B, neti->matrixDim));

  TransitionMatrix<double> C = neti->seq(A, B);
  EXPECT_TRUE(isStochastic(C, neti->matrixDim));
}

TEST_F(NetikatTest, Choice) {
  TransitionMatrix<double> setZeroMat = neti->set(0, 0);
  TransitionMatrix<double> setOneMat = neti->set(0, 1);
  double p = 0.25;
  TransitionMatrix<double> probMat = neti->choice(p, setZeroMat, setOneMat);
  Distribution<double> v = neti->toVec(PacketSet{pkt(1)});
  Eigen::VectorXd v0 = toVec(neti->toVec(PacketSet{pkt(0)}));
  Eigen::VectorXd v1 = toVec(neti->toVec(PacketSet{pkt(1)}));

  Eigen::VectorXd chosenVec = toVec(probMat(v));
  Eigen::VectorXd trueChosenVec = p * v0 + (1 - p) * v1;

  EXPECT_MAT_EQ(chosenVec, trueChosenVec);
}

TEST_F(NetikatTest, StarApproxPreservesStochastic) {
  TransitionMatrix<double> A = randomDenseStochasticOperator(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A, neti->matrixDim));

  TransitionMatrix<double> B = neti->starApprox(A);
  EXPECT_TRUE(isStochastic(B, neti->matrixDim));
}

TEST_F(NetikatTest, StarApproxKindaConverged) {
  double tol = 1e-12;
  TransitionMatrix<double> A = randomDenseStochasticOperator(neti->matrixDim);
  TransitionMatrix<double> approxStar = neti->starApprox(A, tol);
  size_t iter = neti->starIter;
  TransitionMatrix<double> nextApproxStar = neti->dumbStarApprox(A, iter + 1);
  Eigen::SparseMatrix<double> approxStarMat =
      toMat(approxStar, neti->matrixDim);
  Eigen::SparseMatrix<double> nextApproxStarMat =
      toMat(nextApproxStar, neti->matrixDim);

  // TODO: why do I lose precision?
  EXPECT_MAT_NEAR(nextApproxStarMat, approxStarMat, 1e-6);
}
