// unit tests for Simbox class
// feel free to add more

#include <gtest/gtest.h>
#include <memory>

#include <frame.h>
#include <param.h>
#include <simbox.h>

// testing sparsity structure
TEST(simbox, sparsity) {
  unsigned int baseline_int;
  // build two frames
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
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_spatial_frame = std::make_unique<Frame_spatial<3>>(test_par.get());
  test_spatial_frame->init();

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
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_spectral_frame =
      std::make_unique<Frame_spectral<1>>(test_par.get());
  test_spectral_frame->init();

  // check SP non-zero elements
  baseline_int = test_spatial_frame->dsp->n_nonzero_elements() *
                 test_spectral_frame->dsp->n_nonzero_elements();
  auto test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  test_simbox->init();

  EXPECT_EQ(test_simbox->sparsity->n_nonzero_elements(), baseline_int);
  // check product correctness
  // the very first block of simbox DSP must equal to spatial frame DSP
  // however iterators enumerate in arbitrary style
  // we check existness of given column/row positions
  auto it_right = test_spatial_frame->dsp->begin();
  auto end_right = test_spatial_frame->dsp->end();
  for (; it_right != end_right; ++it_right) {
    EXPECT_TRUE(test_simbox->dsp->exists(it_right->row(), it_right->column()));
  }
}

// testing refinement
TEST(simbox, refinement) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1.0;
  test_par->grid_set.q1_min = 0.01;
  test_par->grid_set.nq1 = 5;
  test_par->pip_set.spectral_dim = 1;
  // 1D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 5;
  test_par->pip_set.spatial_dim = 1;
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spatial_max_refine_lv = 3;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.spectral_max_refine_lv = 3;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<1, 1>>(test_par.get());
  test_simbox->init();

  baseline_int = 5;
  EXPECT_EQ(test_simbox->spatial_frame->dof_handler->n_dofs(), baseline_int);
  EXPECT_EQ(test_simbox->spectral_frame->dof_handler->n_dofs(), baseline_int);

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

  EXPECT_NE(test_simbox->spatial_frame->dof_handler->n_dofs(), baseline_int);
  EXPECT_NE(test_simbox->spectral_frame->dof_handler->n_dofs(), baseline_int);

  // check refinement level
  baseline_int = 2;
  EXPECT_EQ(test_simbox->spatial_frame->triangulation->n_levels(),
            baseline_int);
  EXPECT_EQ(test_simbox->spectral_frame->triangulation->n_levels(),
            baseline_int);
}

// testing sparsity structure
TEST(dg_simbox, sparsity) {
  unsigned int baseline_int;
  // build two frames
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
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_spatial_frame = std::make_unique<Frame_spatial<3>>(test_par.get());
  test_spatial_frame->init();

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
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_spectral_frame =
      std::make_unique<Frame_spectral<1>>(test_par.get());
  test_spectral_frame->init();

  // check SP non-zero elements
  baseline_int = test_spatial_frame->dsp->n_nonzero_elements() *
                 test_spectral_frame->dsp->n_nonzero_elements();
  auto test_simbox = std::make_unique<Simbox<3, 1>>(test_par.get());
  test_simbox->init();

  EXPECT_EQ(test_simbox->sparsity->n_nonzero_elements(), baseline_int);
  // check product correctness
  // the very first block of simbox DSP must equal to spatial frame DSP
  // however iterators enumerate in arbitrary style
  // we check existness of given column/row positions
  auto it_right = test_spatial_frame->dsp->begin();
  auto end_right = test_spatial_frame->dsp->end();
  for (; it_right != end_right; ++it_right) {
    EXPECT_TRUE(test_simbox->dsp->exists(it_right->row(), it_right->column()));
  }
}

// testing refinement
TEST(dg_simbox, refinement) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1.0;
  test_par->grid_set.q1_min = 0.01;
  test_par->grid_set.nq1 = 5;
  test_par->pip_set.spectral_dim = 1;
  // 1D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 5;
  test_par->pip_set.spatial_dim = 1;
  // refinement
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spatial_max_refine_lv = 3;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.spectral_max_refine_lv = 3;
  test_par->grid_set.refine_ratio = 1; // full refinement ratio
  test_par->grid_set.coarsen_ratio = 0;

  auto test_simbox = std::make_unique<Simbox<1, 1>>(test_par.get());
  test_simbox->init();

  baseline_int = 5;
  EXPECT_EQ(test_simbox->spatial_frame->dof_handler->n_dofs(), baseline_int);
  baseline_int = 5;
  EXPECT_EQ(test_simbox->spectral_frame->dof_handler->n_dofs(), baseline_int);

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

  EXPECT_NE(test_simbox->spatial_frame->dof_handler->n_dofs(), baseline_int);
  EXPECT_NE(test_simbox->spectral_frame->dof_handler->n_dofs(), baseline_int);

  // check refinement level
  baseline_int = 2;
  EXPECT_EQ(test_simbox->spatial_frame->triangulation->n_levels(),
            baseline_int);
  EXPECT_EQ(test_simbox->spectral_frame->triangulation->n_levels(),
            baseline_int);
}
