// unit tests for system class
// feel free to add more

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>

#include <param.h>
#include <simbox.h>
#include <system.h>

// testing init matrix size
TEST(system, init) {
  unsigned int baseline_int;
  // build frames
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
  // build base operator obj (mass matrices)
  auto test_sys = std::make_unique<System<3, 1>>(test_par.get());
  test_sys->init(test_simbox.get());
  // check Mx and Mq size
  EXPECT_EQ(test_sys->Mx->m(), test_sys->Mx->n());
  EXPECT_EQ(test_sys->Mq->m(), test_sys->Mq->n());
  baseline_int = 27;
  EXPECT_EQ(test_sys->Rx->size(), baseline_int);
  EXPECT_EQ(test_sys->Mx->m(), baseline_int);
  baseline_int = 3;
  EXPECT_EQ(test_sys->Rq->size(), baseline_int);
  EXPECT_EQ(test_sys->Mq->m(), baseline_int);
  // check Mxq size
  baseline_int = 3 * 27;
  EXPECT_EQ(test_sys->Rxq->size(), baseline_int);
  EXPECT_EQ(test_sys->Mxq->m(), test_sys->Mxq->n());
  EXPECT_EQ(test_sys->Mxq->m(), baseline_int);
}

TEST(system, mass_matrix) {
  // build frames
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
  // build base operator obj (mass matrices)
  auto test_sys = std::make_unique<System<3, 1>>(test_par.get());
  test_sys->init(test_simbox.get());
  test_sys->assemble_mass_Mxq(test_simbox.get());
  // compare Mxq and mass_Mxq, they should equal
  EXPECT_EQ(test_sys->Mxq->m(), test_sys->mass_Mxq->m());
  for (unsigned int i = 0; i < test_sys->Mxq->m(); ++i) {
    for (unsigned int j = 0; j < test_sys->Mxq->n(); ++j) {
      EXPECT_TRUE(fabs(test_sys->Mxq->el(i, j) - test_sys->mass_Mxq->el(i, j)) <
                  1.e-5);
    }
  }
}

// testing refinement matrix size
TEST(system, refine) {
  unsigned int baseline_int;
  // build frames
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
  test_par->grid_set.spatial_max_refine_lv = 3;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.spectral_max_refine_lv = 3;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  test_simbox->init();
  // build base operator obj (mass matrices)
  auto test_sys = std::make_unique<System<3, 1>>(test_par.get());
  test_sys->init(test_simbox.get());

  // refine simbox
  dealii::Vector<float> test_spa_err, test_spe_err;
  test_spa_err.reinit(
      test_simbox->spatial_frame->triangulation->n_active_cells());
  test_spe_err.reinit(
      test_simbox->spectral_frame->triangulation->n_active_cells());
  for (decltype(test_spa_err.size()) i = 0; i < test_spa_err.size(); ++i)
    test_spa_err[i] = i * 0.01;
  for (decltype(test_spe_err.size()) i = 0; i < test_spe_err.size(); ++i)
    test_spe_err[i] = i * 0.01;

  test_simbox->refine(&test_spa_err, &test_spe_err);
  test_sys->refine(test_simbox.get());

  // check Mxq size
  baseline_int = 3 * 27;
  EXPECT_NE(test_sys->Rxq->size(), baseline_int);
  EXPECT_NE(test_sys->Mxq->m(), baseline_int);
}

// test constant RHS
TEST(system, const_rhs) {
  // build frames
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
  // build base operator obj (mass matrices)
  auto test_sys = std::make_unique<System<3, 1>>(test_par.get());
  test_sys->init(test_simbox.get());
  test_sys->assemble_const_Rxq(test_simbox.get());
  // compare Rxq and const_Rxq, they should equal
  EXPECT_EQ(test_sys->Rxq->size(), test_sys->const_Rxq->size());
  for (unsigned int i = 0; i < test_sys->Rxq->size(); ++i) {
    std::cout << i << std::endl;
    EXPECT_TRUE(fabs((*(test_sys->Rxq))[i] - (*(test_sys->const_Rxq))[i]) <
                1.e-5);
  }
}
