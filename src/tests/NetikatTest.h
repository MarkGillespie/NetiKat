#include "netikat.h"
#include "utils.h"

#include <bitset>

class NetikatTest : public testing::Test {
public:
  static std::unique_ptr<NetiKAT> neti;

protected:
  static void SetUpTestSuite() {
    std::vector<size_t> packetType{2, 2};
    neti = std::unique_ptr<NetiKAT>(new NetiKAT(packetType, 4));
  }

  void SetUp() override {}

  static void TearDownTestSuite() { neti.release(); }

  static inline size_t pkt(size_t a);
  static inline size_t pkt(size_t a, size_t b);
};

std::unique_ptr<NetiKAT> NetikatTest::neti = nullptr;

size_t NetikatTest::pkt(size_t a) {
  return neti->packetIndex(std::vector<size_t>{a, 0});
}
size_t NetikatTest::pkt(size_t a, size_t b) {
  return neti->packetIndex(std::vector<size_t>{a, b});
}

// ============================================
//        TransitionMatrix Tests
// ============================================
TEST_F(NetikatTest, binomialCoefficients) {
  size_t nCk = binom(56, 12);
  size_t answer = 558383307300; // Solution computed in Mathematica

  EXPECT_EQ(nCk, answer);
}

TEST_F(NetikatTest, normalizedMatrixIsStochastic) {
  size_t dim = 10;
  TransitionMatrix A = randomPositiveSparse(dim, 0.5);

  neti->normalize(A);
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(dim);
  Eigen::VectorXd colSums = A.transpose() * ones;

  EXPECT_MAT_NEAR(colSums, ones, 1e-8);
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

TEST_F(NetikatTest, bigIndexUnindexInverse) {
  size_t i, j;
  for (size_t iP = 0; iP < neti->matrixDim * neti->matrixDim; ++iP) {
    std::tie(i, j) = neti->bigUnindex(iP);
    size_t newIndex = neti->bigIndex(i, j);
    ASSERT_EQ(iP, newIndex);
  }
}

TEST_F(NetikatTest, Skip) {
  TransitionMatrix skipMat = neti->skip();

  // Generate a random probability distribution
  Eigen::VectorXd v = Eigen::VectorXd::Random(neti->matrixDim);
  for (size_t i = 0; i < neti->matrixDim; ++i) {
    v(i) = abs(v(i));
  }
  v /= v.lpNorm<1>();

  Eigen::VectorXd skipV = skipMat * v;

  EXPECT_MAT_EQ(v, skipV);
}

TEST_F(NetikatTest, Drop) {
  TransitionMatrix dropMat = neti->drop();
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(1)});

  Eigen::VectorXd droppedVec = dropMat * v;

  Eigen::VectorXd trueDroppedVec = neti->toVec(PacketSet());

  EXPECT_MAT_EQ(droppedVec, trueDroppedVec);
}

TEST_F(NetikatTest, Test) {
  TransitionMatrix testMat = neti->test(0, 1);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(0), pkt(1)});
  Eigen::VectorXd testedVec = testMat * v;

  Eigen::VectorXd trueTestedVec = neti->toVec(PacketSet{pkt(1)});

  EXPECT_MAT_EQ(testedVec, trueTestedVec);
}

TEST_F(NetikatTest, TestSize) {
  TransitionMatrix testMat = neti->testSize(0, 1, 2);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(1, 1), pkt(1, 0)});
  Eigen::VectorXd testedVec = testMat * v;

  EXPECT_MAT_EQ(testedVec, v);
}

TEST_F(NetikatTest, Set) {
  TransitionMatrix setMat = neti->set(0, 1);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(0)});
  Eigen::VectorXd setVec = setMat * v;

  Eigen::VectorXd trueSetVec = neti->toVec(PacketSet{pkt(1)});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, Amp) {
  TransitionMatrix setZeroMat = neti->set(0, 0);
  TransitionMatrix setOneMat = neti->set(0, 1);
  TransitionMatrix setBothMat = neti->amp(setZeroMat, setOneMat);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(0)});
  Eigen::VectorXd setVec = setBothMat * v;

  Eigen::VectorXd trueSetVec = neti->toVec(PacketSet{pkt(0), pkt(1)});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, AmpPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  TransitionMatrix B = randomDenseStochastic(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A));
  EXPECT_TRUE(isStochastic(B));

  TransitionMatrix C = neti->amp(A, B);
  EXPECT_TRUE(isStochastic(C));
}

TEST_F(NetikatTest, AmpCommutes) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  TransitionMatrix B = randomDenseStochastic(neti->matrixDim);

  TransitionMatrix C = neti->amp(A, B);
  TransitionMatrix D = neti->amp(B, A);
  EXPECT_MAT_NEAR(C, D, 1e-12);
}

TEST_F(NetikatTest, Seq) {
  TransitionMatrix setZeroMat = neti->set(0, 0);
  TransitionMatrix setOneMat = neti->set(0, 1);
  TransitionMatrix setZeroThenOneMat = neti->seq(setOneMat, setZeroMat);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(1)});

  Eigen::VectorXd setZeroVec = setZeroMat * v;
  Eigen::VectorXd trueSetZeroVec = neti->toVec(PacketSet{pkt(0)});
  EXPECT_MAT_EQ(setZeroVec, trueSetZeroVec);

  Eigen::VectorXd setVec = setZeroThenOneMat * v;
  Eigen::VectorXd trueSetVec = neti->toVec(PacketSet{pkt(1)});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(NetikatTest, SeqPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  TransitionMatrix B = randomDenseStochastic(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A));
  EXPECT_TRUE(isStochastic(B));

  TransitionMatrix C = neti->seq(A, B);
  EXPECT_TRUE(isStochastic(C));
}

TEST_F(NetikatTest, Choice) {
  TransitionMatrix setZeroMat = neti->set(0, 0);
  TransitionMatrix setOneMat = neti->set(0, 1);
  double p = 0.25;
  TransitionMatrix probMat = neti->choice(p, setZeroMat, setOneMat);
  Eigen::VectorXd v = neti->toVec(PacketSet{pkt(1)});
  Eigen::VectorXd v0 = neti->toVec(PacketSet{pkt(0)});
  Eigen::VectorXd v1 = neti->toVec(PacketSet{pkt(1)});

  Eigen::VectorXd chosenVec = probMat * v;
  Eigen::VectorXd trueChosenVec = p * v0 + (1 - p) * v1;

  EXPECT_MAT_EQ(chosenVec, trueChosenVec);
}

TEST_F(NetikatTest, StarPreservesStochastic) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  EXPECT_TRUE(isStochastic(A));

  TransitionMatrix B = neti->star(A);
  EXPECT_TRUE(isStochastic(B));
}

TEST_F(NetikatTest, StarAgreesWithStarApprox) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  TransitionMatrix star = neti->star(A);
  TransitionMatrix approxStar = neti->starApprox(A, 1e-12);

  EXPECT_MAT_NEAR(star, approxStar, 1e-12);
}

TEST_F(NetikatTest, StarApproxCountsIterationsCorrectly) {
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  size_t iter;
  TransitionMatrix approxStar = neti->starApprox(A, 1e-5, iter);
  TransitionMatrix iterStar = neti->dumbStarApprox(A, iter);

  EXPECT_MAT_NEAR(iterStar, approxStar, 1e-12);
}

TEST_F(NetikatTest, StarApproxKindaConverged) {
  double tol = 1e-12;
  TransitionMatrix A = randomDenseStochastic(neti->matrixDim);
  size_t iter;
  TransitionMatrix approxStar = neti->starApprox(A, tol, iter);
  TransitionMatrix nextApproxStar = neti->dumbStarApprox(A, iter + 1);

  EXPECT_MAT_NEAR(nextApproxStar, approxStar, tol);
}
