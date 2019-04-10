// unit tests for advection classes
// feel free to add mroe

#include <cmath>
#include <gtest/gtest.h>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>

#include <advection.h>

// checking tensor product useage
TEST(advection, tensor_product) {
  auto A = dealii::Tensor<2, 3, double>();
  auto x = dealii::Tensor<1, 3, double>({2, 1, 3});
  auto y = dealii::Tensor<1, 3, double>({0, 1, 2});
  A[dealii::TableIndices<2>(0, 0)] = 1;
  A[dealii::TableIndices<2>(1, 1)] = 1;
  A[dealii::TableIndices<2>(2, 2)] = 2;
  double test_val = 13.;
  EXPECT_EQ(dealii::scalar_product(y, A * x), test_val);
}

// base class
TEST(advection, base_class) {
  // R32
  auto test_advec = std::make_unique<Advection<3, 2>>();
  auto test_Axx = test_advec->Axx(dealii::Point<3, double>(1, 2, 3),
                                  dealii::Point<2, double>(4, 5));
  auto test_Aqq = test_advec->Aqq(dealii::Point<3, double>(1, 2, 3),
                                  dealii::Point<2, double>(4, 5));
  dealii::Tensor<1, 3, double> baseline_Axx;
  dealii::Tensor<1, 2, double> baseline_Aqq;
  baseline_Axx = 0;
  baseline_Aqq = 0;
  EXPECT_EQ(test_Axx, baseline_Axx);
  EXPECT_EQ(test_Aqq, baseline_Aqq);
}
