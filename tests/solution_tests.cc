// unit tests for Solutions class
// feel free to add more

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <param.h>
#include <simbox.h>
#include <solution.h>

// testing reshaping solution size
TEST(solution, reshape) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();
  // 3D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  test_par->pip_set.spatial_dim = 3;
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.q2_max = 0;
  test_par->grid_set.q2_min = -2;
  test_par->grid_set.q3_max = 0;
  test_par->grid_set.q3_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  test_par->pip_set.spectral_dim = 1;
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  test_simbox->init();
  auto test_sols = std::make_unique<Solution<3, 1>>(test_par.get());
  // not invoking Solution::init

  baseline_int = 81; // 27*3
  test_sols->new_reshape(test_simbox.get());
  test_sols->old_reshape(test_simbox.get());
  EXPECT_EQ(test_sols->Snew->size(), baseline_int);
  EXPECT_EQ(test_sols->Sold->size(), baseline_int);

  // resized 3D spatial
  test_par->grid_set.nx1 = 2;
  test_par->grid_set.nx2 = 2;
  test_par->grid_set.nx3 = 2;
  auto new_test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  new_test_simbox->init();

  baseline_int = 24; // 8*3
  test_sols->new_reshape(new_test_simbox.get());
  test_sols->old_reshape(new_test_simbox.get());
  EXPECT_EQ(test_sols->Snew->size(), baseline_int);
  EXPECT_EQ(test_sols->Sold->size(), baseline_int);
  baseline_int = 8;
  EXPECT_EQ(test_sols->n_rows_old, baseline_int);
  EXPECT_EQ(test_sols->n_rows_new, baseline_int);
  baseline_int = 3;
  EXPECT_EQ(test_sols->n_cols_old, baseline_int);
  EXPECT_EQ(test_sols->n_cols_new, baseline_int);
}

// testing element picking in solution "matrix U"
TEST(solution, elements) {
  const double baseline_double{45.02};
  auto test_par = std::make_unique<Param>();
  // 3D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  test_par->pip_set.spatial_dim = 3;
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.q2_max = 0;
  test_par->grid_set.q2_min = -2;
  test_par->grid_set.q3_max = 0;
  test_par->grid_set.q3_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  test_par->pip_set.spectral_dim = 1;
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  test_simbox->init();

  auto test_sols = std::make_unique<Solution<3, 1>>(test_par.get());
  // not invoking Solution::init
  test_sols->new_reshape(test_simbox.get());
  test_sols->old_reshape(test_simbox.get());

  const unsigned test_row_idx = 23;
  const unsigned test_col_idx = 2;
  test_sols->new_el(test_row_idx, test_col_idx) = baseline_double;
  EXPECT_EQ(test_sols->new_el(test_row_idx, test_col_idx), baseline_double);
  test_sols->old_el(test_row_idx, test_col_idx) = baseline_double;
  EXPECT_EQ(test_sols->old_el(test_row_idx, test_col_idx), baseline_double);
}

// this is a special unit test
// which tests both "evaluate" and "_solution_init" functions
TEST(solution, init_evaluate) {
  double baseline_double;
  auto test_par = std::make_unique<Param>();
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->pip_set.spectral_dim = 1;
  // 1D spatial global
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->pip_set.spatial_dim = 1;
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 3;
  test_par->grid_set.spectral_min_refine_lv = 8;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<1, 1>>(test_par.get());
  test_simbox->init();
  auto test_sols = std::make_unique<Solution<1, 1>>(test_par.get());
  test_sols->init(test_simbox.get());
  test_sols->pre_constraints(test_simbox.get());
  test_sols->post_constraints(test_simbox.get());

  auto test_spatial_inc = std::make_unique<Solution<1, 1>::Spatial_initial>();
  auto test_spectral_inc = std::make_unique<Solution<1, 1>::Spectral_initial>();

  // evaluate solution
  for (auto i : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    dealii::Point<1, double> test_point_spatial(i);
    for (auto j : {-0.1, -0.4, -0.7, -0.9}) {
      dealii::Point<1, double> test_point_spectral(j);
      baseline_double = test_spatial_inc->value(test_point_spatial) *
                        test_spectral_inc->value(test_point_spectral);
      EXPECT_EQ(test_sols->evaluate(test_simbox.get(), test_point_spatial,
                                    test_point_spectral),
                baseline_double);
    }
    baseline_double = 0.; // default boundary
    dealii::Point<1, double> test_point_spectral_upper(
        test_par->grid_set.q1_max);
    EXPECT_EQ(test_sols->evaluate(test_simbox.get(), test_point_spatial,
                                  test_point_spectral_upper),
              baseline_double);
    dealii::Point<1, double> test_point_spectral_lower(
        test_par->grid_set.q1_min);
    EXPECT_EQ(test_sols->evaluate(test_simbox.get(), test_point_spatial,
                                  test_point_spectral_lower),
              baseline_double);
  }
}

// testing refinement
TEST(solution, refinement) {
  auto test_par = std::make_unique<Param>();
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->pip_set.spectral_dim = 1;
  // 1D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->pip_set.spatial_dim = 1;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 6;
  test_par->grid_set.spatial_max_refine_lv = 10;
  test_par->grid_set.spectral_min_refine_lv = 6;
  test_par->grid_set.spectral_max_refine_lv = 10;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_crudesimbox = std::make_unique<Simbox<1, 1>>(test_par.get());
  test_crudesimbox->init();
  auto test_crudesols = std::make_unique<Solution<1, 1>>(test_par.get());
  test_crudesols->init(test_crudesimbox.get());
  auto test_finesimbox = std::make_unique<Simbox<1, 1>>(test_par.get());
  test_finesimbox->init();
  auto test_finesols = std::make_unique<Solution<1, 1>>(test_par.get());
  test_finesols->init(test_finesimbox.get());

  // refine solution
  dealii::Vector<float> test_spa_err, test_spe_err;
  test_spa_err.reinit(
      test_finesimbox->spatial_frame->triangulation->n_active_cells());
  test_spe_err.reinit(
      test_finesimbox->spectral_frame->triangulation->n_active_cells());
  for (decltype(test_spa_err.size()) i = 0; i < test_spa_err.size(); ++i)
    test_spa_err[i] = i * 0.01;
  for (decltype(test_spe_err.size()) i = 0; i < test_spe_err.size(); ++i)
    test_spe_err[i] = i * 0.01;

  // spatial domain adaptive refine
  test_finesols->refine_spatial(test_finesimbox.get(), &test_spa_err);
  // evaluate solution
  for (auto i : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    for (auto j : {-0.1, -0.4, -0.7, -0.9}) {
      dealii::Point<1, double> test_point_spatial(i);
      dealii::Point<1, double> test_point_spectral(j);

      // std::cout<<"testing value: "<<test_crudesols->evaluate
      // (test_crudesimbox.get(),
      //                                                        test_point_spatial,
      //                                                        test_point_spectral)
      //<<std::endl;

      EXPECT_TRUE(fabs(test_crudesols->evaluate(test_crudesimbox.get(),
                                                test_point_spatial,
                                                test_point_spectral) -
                       test_finesols->evaluate(test_finesimbox.get(),
                                               test_point_spatial,
                                               test_point_spectral)) < 1.e-5);
    }
  }

  // spectral doamin adaptive refine
  test_finesols->refine_spectral(test_finesimbox.get(), &test_spe_err);
  // evaluate solution
  for (auto i : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    for (auto j : {-0.1, -0.4, -0.7, -0.9}) {
      dealii::Point<1, double> test_point_spatial(i);
      dealii::Point<1, double> test_point_spectral(j);

      // std::cout<<"testing value: "<<test_crudesols->evaluate
      // (test_crudesimbox.get(),
      //                                                        test_point_spatial,
      //                                                        test_point_spectral)
      //<<std::endl;

      EXPECT_TRUE(fabs(test_crudesols->evaluate(test_crudesimbox.get(),
                                                test_point_spatial,
                                                test_point_spectral) -
                       test_finesols->evaluate(test_finesimbox.get(),
                                               test_point_spatial,
                                               test_point_spectral)) < 1.e-5);
    }
  }

  // global refine
  test_finesols->refine(test_finesimbox.get(),
                        (const dealii::Vector<float> *)nullptr,
                        (const dealii::Vector<float> *)nullptr);
  // evaluate solution
  for (auto i : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    for (auto j : {-0.1, -0.4, -0.7, -0.9}) {
      dealii::Point<1, double> test_point_spatial(i);
      dealii::Point<1, double> test_point_spectral(j);

      // std::cout<<"testing value: "<<test_crudesols->evaluate
      // (test_crudesimbox.get(),
      //                                                        test_point_spatial,
      //                                                        test_point_spectral)
      //<<std::endl;

      EXPECT_TRUE(fabs(test_crudesols->evaluate(test_crudesimbox.get(),
                                                test_point_spatial,
                                                test_point_spectral) -
                       test_finesols->evaluate(test_finesimbox.get(),
                                               test_point_spatial,
                                               test_point_spectral)) < 1.e-5);
    }
  }

  // adaptive-global refine
  test_crudesols->refine(test_crudesimbox.get(),
                         (const dealii::Vector<float> *)nullptr, &test_spa_err);
  // evaluate solution
  for (auto i : {0.2, 0.3, 0.4, 0.5, 0.6}) {
    for (auto j : {-0.1, -0.4, -0.7, -0.9}) {
      dealii::Point<1, double> test_point_spatial(i);
      dealii::Point<1, double> test_point_spectral(j);

      // std::cout<<"testing value: "<<test_crudesols->evaluate
      // (test_crudesimbox.get(),
      //                                                        test_point_spatial,
      //                                                        test_point_spectral)
      //<<std::endl;

      EXPECT_TRUE(fabs(test_crudesols->evaluate(test_crudesimbox.get(),
                                                test_point_spatial,
                                                test_point_spectral) -
                       test_finesols->evaluate(test_finesimbox.get(),
                                               test_point_spatial,
                                               test_point_spectral)) < 1.e-5);
    }
  }
}
