// unit tests for diffusion classes
// feel free to add mroe

#include <cmath>
#include <gtest/gtest.h>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>

#include <diffusion.h>

// base class
TEST(diffusion, base_class) {
  // R13
  auto test_diff = std::make_unique<Diffusion<1, 3>>();
  auto test_Dxx = test_diff->Dxx(dealii::Point<1, double>(1),
                                 dealii::Point<3, double>(2, 3, 4));
  auto test_Dqq = test_diff->Dqq(dealii::Point<1, double>(1),
                                 dealii::Point<3, double>(2, 3, 4));
  dealii::Tensor<2, 1, double> baseline_Dxx;
  dealii::Tensor<2, 3, double> baseline_Dqq;
  baseline_Dxx = 0;
  baseline_Dqq = 0;
  EXPECT_EQ(test_Dxx, baseline_Dxx);
  EXPECT_EQ(test_Dqq, baseline_Dqq);
}

// END
