#include "TransitionMatrix.h"

class TransitionMatrixTest : public testing::Test {
public:
  static std::unique_ptr<PacketSet> set;

protected:
  static void SetUpTestSuite() {
    std::vector<size_t> packetType{2, 2};
    set = std::unique_ptr<PacketSet>(new PacketSet(packetType));
  }

  void SetUp() override {}

  static void TearDownTestSuite() { set.release(); }
};

std::unique_ptr<PacketSet> TransitionMatrixTest::set = nullptr;

// ============================================
//        TransitionMatrix Tests
// ============================================
TEST_F(TransitionMatrixTest, indexOfPacketFromIndexIsIdentity) {
  for (size_t iP = 0; iP < set->possiblePackets; ++iP) {
    size_t newIndex = set->packetIndex(set->packetFromIndex(iP));
    EXPECT_EQ(iP, newIndex);
  }
}

TEST_F(TransitionMatrixTest, indexOfPacketSetFromIndexIsIdentity) {
  for (size_t iP = 0; iP < set->matrixDim; ++iP) {
    size_t newIndex = set->index(set->packetSetFromIndex(iP));
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
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1, 1}});
  Eigen::VectorXd droppedVec = dropMat * v;

  Eigen::VectorXd trueDroppedVec = set->toVec(std::set<Packet>());

  EXPECT_MAT_EQ(droppedVec, trueDroppedVec);
}

TEST_F(TransitionMatrixTest, Test) {
  TransitionMatrix testMat = set->test(0, 1);
  Eigen::VectorXd v = set->toVec(
      std::set<Packet>{std::vector<size_t>{0, 0}, std::vector<size_t>{1, 0}});
  Eigen::VectorXd testedVec = testMat * v;

  Eigen::VectorXd trueTestedVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});

  EXPECT_MAT_EQ(testedVec, trueTestedVec);
}

TEST_F(TransitionMatrixTest, TestSize) {
  TransitionMatrix testMat = set->testSize(0, 1, 2);
  Eigen::VectorXd v = set->toVec(
      std::set<Packet>{std::vector<size_t>{1, 1}, std::vector<size_t>{1, 0}});
  Eigen::VectorXd testedVec = testMat * v;

  EXPECT_MAT_EQ(testedVec, v);
}

TEST_F(TransitionMatrixTest, Set) {
  TransitionMatrix setMat = set->set(0, 1);
  Eigen::VectorXd v = set->toVec(
      std::set<Packet>{std::vector<size_t>{0, 0}, std::vector<size_t>{0, 1}});
  Eigen::VectorXd setVec = setMat * v;

  Eigen::VectorXd trueSetVec = set->toVec(
      std::set<Packet>{std::vector<size_t>{1, 0}, std::vector<size_t>{1, 1}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, Amp) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  TransitionMatrix setBothMat = set->amp(setZeroMat, setOneMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{0, 0}});
  Eigen::VectorXd setVec = setBothMat * v;

  Eigen::VectorXd trueSetVec = set->toVec(
      std::set<Packet>{std::vector<size_t>{0, 0}, std::vector<size_t>{1, 0}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, Seq) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  TransitionMatrix setZeroThenOneMat = set->seq(setOneMat, setZeroMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});

  Eigen::VectorXd setZeroVec = setZeroMat * v;
  Eigen::VectorXd trueSetZeroVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{0, 0}});
  EXPECT_MAT_EQ(setZeroVec, trueSetZeroVec);

  Eigen::VectorXd setVec = setZeroThenOneMat * v;
  Eigen::VectorXd trueSetVec =
      set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});

  EXPECT_MAT_EQ(setVec, trueSetVec);
}

TEST_F(TransitionMatrixTest, Choice) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix setOneMat = set->set(0, 1);
  double p = 0.25;
  TransitionMatrix probMat = set->choice(p, setZeroMat, setOneMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});
  Eigen::VectorXd v0 = set->toVec(std::set<Packet>{std::vector<size_t>{0, 0}});
  Eigen::VectorXd v1 = set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});

  Eigen::VectorXd chosenVec = probMat * v;
  Eigen::VectorXd trueChosenVec = p * v0 + (1 - p) * v1;

  EXPECT_MAT_EQ(chosenVec, trueChosenVec);
}

TEST_F(TransitionMatrixTest, Star) {
  TransitionMatrix setZeroMat = set->set(0, 0);
  TransitionMatrix starMat = set->star(setZeroMat);
  Eigen::VectorXd v = set->toVec(std::set<Packet>{std::vector<size_t>{1, 0}});

  Eigen::VectorXd starVec = starMat * v;

  Eigen::VectorXd trueStarVec = set->toVec(
      std::set<Packet>{std::vector<size_t>{0, 0}, std::vector<size_t>{1, 0}});

  // TODO: not exact?
  EXPECT_MAT_NEAR(starVec, trueStarVec, 1e-12);
}
