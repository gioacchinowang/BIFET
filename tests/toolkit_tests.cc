// unit tests for namespace toolkit
// feel free to add more

#include <gtest/gtest.h>

#include <deal.II/lac/vector.h>
#include <memory>
#include <vector>

#include <namespace_toolkit.h>

// testing general vector Kronecker product
TEST(toolkit, Kronecker_product) {
  // templated by std::vector
  auto test_std_col =
      std::make_unique<std::vector<double>>(std::vector<double>{1.0, 2.0});
  auto test_std_row =
      std::make_unique<std::vector<double>>(std::vector<double>{1.0, 2.0, 3.0});
  auto test_std_rslt = std::make_unique<std::vector<double>>(6, 0.0);
  // output of cross_prod should be in Fortran style
  auto test_std_val = std::make_unique<std::vector<double>>(
      std::vector<double>{1.0, 2.0, 2.0, 4.0, 3.0, 6.0});

  toolkit::Kronecker_product<std::vector<double>>(
      test_std_row.get(), test_std_col.get(), test_std_rslt.get());

  EXPECT_EQ(test_std_rslt->size(), test_std_val->size());
  for (decltype(test_std_val->size()) i = 0; i < test_std_val->size(); ++i)
    EXPECT_EQ((*test_std_rslt)[i], (*test_std_val)[i]);
  // templated by dealii::Vector
  auto test_deal_col = std::make_unique<dealii::Vector<double>>(2);
  auto test_deal_row = std::make_unique<dealii::Vector<double>>(3);
  for (auto i = 0; i < 2; ++i)
    (*test_deal_col)[i] = i + 1.0;
  for (auto i = 0; i < 3; ++i)
    (*test_deal_row)[i] = i + 1.0;
  auto test_deal_rslt = std::make_unique<dealii::Vector<double>>(6);

  toolkit::Kronecker_product<dealii::Vector<double>>(
      test_deal_row.get(), test_deal_col.get(), test_deal_rslt.get());

  EXPECT_EQ(test_deal_rslt->size(), test_std_val->size());
  for (decltype(test_std_val->size()) i = 0; i < test_std_val->size(); ++i)
    EXPECT_EQ((*test_deal_rslt)[i], (*test_std_val)[i]);
}
