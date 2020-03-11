#include "test_utils.h"

#include <gtest/gtest.h>

// clang-format off
#include "TransitionMatrixTest.h"
// clang-format on

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
