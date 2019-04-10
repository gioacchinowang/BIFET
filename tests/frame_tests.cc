// unit tests for Frames class and derived classes
// feel free to add more

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include <frame.h>
#include <param.h>

// testing spatial triangulation
TEST(frame, spatial_triangulation) {
  unsigned int baseline_int;
  std::vector<unsigned int> baseline_vec;
  dealii::Point<1, double> baseline_point;
  auto test_par = std::make_unique<Param>();
  // general grid settings
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
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  // 1D spatial
  test_par->pip_set.spatial_dim = 1;
  auto test_frame1 = std::make_unique<Frame_spatial<1>>(test_par.get());
  test_frame1->init();
  // check active cells
  baseline_int = 2;
  EXPECT_EQ(test_frame1->triangulation->n_active_cells(), baseline_int);
  // check dofs
  baseline_int = 3;
  EXPECT_EQ(test_frame1->dof_handler->n_dofs(), baseline_int);
  // check polynomial order
  baseline_int = 1;
  EXPECT_EQ(test_frame1->pol_order, baseline_int);
  // check min refine lv
  baseline_int = 0;
  EXPECT_EQ(test_frame1->min_refine_lv, baseline_int);
  // check block number
  baseline_vec.push_back(2);
  EXPECT_EQ(test_frame1->block_nums, baseline_vec);
  // check pivot points
  baseline_point = dealii::Point<1, double>(0);
  EXPECT_EQ(test_frame1->pivot_min, baseline_point);
  baseline_point = dealii::Point<1, double>(1);
  EXPECT_EQ(test_frame1->pivot_max, baseline_point);
  // check sparsity
  EXPECT_EQ(test_frame1->dsp->n_nonzero_elements(),
            test_frame1->sparsity->n_nonzero_elements());

  // 2D spatial
  test_par->pip_set.spatial_dim = 2;
  auto test_frame2 = std::make_unique<Frame_spatial<2>>(test_par.get());
  test_frame2->init();
  baseline_int = 4;
  EXPECT_EQ(test_frame2->triangulation->n_active_cells(), baseline_int);
  baseline_int = 9;
  EXPECT_EQ(test_frame2->dof_handler->n_dofs(), baseline_int);

  // 3D spatial
  test_par->pip_set.spatial_dim = 3;
  auto test_frame3 = std::make_unique<Frame_spatial<3>>(test_par.get());
  test_frame3->init();
  baseline_int = 8;
  EXPECT_EQ(test_frame3->triangulation->n_active_cells(), baseline_int);
  baseline_int = 27;
  EXPECT_EQ(test_frame3->dof_handler->n_dofs(), baseline_int);
}

// testing spectral triangulation
TEST(frames, spectral_triangulation) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();

  // general grid settings
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
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  // 1D spectral
  test_par->pip_set.spectral_dim = 1;
  auto test_frame1 = std::make_unique<Frame_spectral<1>>(test_par.get());
  test_frame1->init();
  baseline_int = 2;
  EXPECT_EQ(test_frame1->triangulation->n_active_cells(), baseline_int);
  baseline_int = 3;
  EXPECT_EQ(test_frame1->dof_handler->n_dofs(), baseline_int);

  // 2D spectral
  test_par->pip_set.spectral_dim = 2;
  auto test_frame2 = std::make_unique<Frame_spectral<2>>(test_par.get());
  test_frame2->init();
  baseline_int = 4;
  EXPECT_EQ(test_frame2->triangulation->n_active_cells(), baseline_int);
  baseline_int = 9;
  EXPECT_EQ(test_frame2->dof_handler->n_dofs(), baseline_int);

  // 3D spectral
  test_par->pip_set.spectral_dim = 3;
  auto test_frame3 = std::make_unique<Frame_spectral<3>>(test_par.get());
  test_frame3->init();
  baseline_int = 8;
  EXPECT_EQ(test_frame3->triangulation->n_active_cells(), baseline_int);
  baseline_int = 27;
  EXPECT_EQ(test_frame3->dof_handler->n_dofs(), baseline_int);
}

// testing frame refinement
TEST(frames, refinement) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();

  test_par->pip_set.spatial_dim = 2;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 2;
  test_par->grid_set.nx2 = 2;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.spatial_min_refine_lv = 2;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_frame = std::make_unique<Frame_spatial<2>>(test_par.get());
  test_frame->init();

  baseline_int = 4;
  EXPECT_EQ(test_frame->max_refine_lv, baseline_int);
  baseline_int = 3;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  baseline_int = 16;
  EXPECT_EQ(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 25;
  EXPECT_EQ(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_EQ(test_frame->sparsity->n_cols(), baseline_int);

  // global refine
  test_frame->refine((const dealii::Vector<float> *)nullptr);

  // refine lv must be +1
  baseline_int = 4;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  baseline_int = 64;
  EXPECT_EQ(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 81;
  EXPECT_EQ(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_EQ(test_frame->sparsity->n_cols(), baseline_int);

  // adaptive refine
  dealii::Vector<float> test_err;
  test_err.reinit(64);
  for (decltype(test_err.size()) i = 0; i < test_err.size(); ++i)
    test_err[i] = i * 0.01;

  test_frame->refine(&test_err);

  // refine lv must be +1
  baseline_int = 5;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  // refined objects should not equal to old ones
  baseline_int = 64;
  EXPECT_NE(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 81;
  EXPECT_NE(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_NE(test_frame->sparsity->n_cols(), baseline_int);
}

// testing spatial triangulation
TEST(dg_frame, spatial_triangulation) {
  unsigned int baseline_int;
  std::vector<unsigned int> baseline_vec;
  dealii::Point<1, double> baseline_point;
  auto test_par = std::make_unique<Param>();

  // general grid settings
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
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  // 1D spatial
  test_par->pip_set.spatial_dim = 1;
  auto test_frame1 = std::make_unique<dGFrame_spatial<1>>(test_par.get());
  test_frame1->init();
  // check active cells
  baseline_int = 2;
  EXPECT_EQ(test_frame1->triangulation->n_active_cells(), baseline_int);
  // check dofs
  baseline_int = 4;
  EXPECT_EQ(test_frame1->dof_handler->n_dofs(), baseline_int);
  // check polynomial order
  baseline_int = 1;
  EXPECT_EQ(test_frame1->pol_order, baseline_int);
  // check min refine lv
  baseline_int = 0;
  EXPECT_EQ(test_frame1->min_refine_lv, baseline_int);
  // check block number
  baseline_vec.push_back(2);
  EXPECT_EQ(test_frame1->block_nums, baseline_vec);
  // check pivot points
  baseline_point = dealii::Point<1, double>(0);
  EXPECT_EQ(test_frame1->pivot_min, baseline_point);
  baseline_point = dealii::Point<1, double>(1);
  EXPECT_EQ(test_frame1->pivot_max, baseline_point);
  // check sparsity
  EXPECT_EQ(test_frame1->dsp->n_nonzero_elements(),
            test_frame1->sparsity->n_nonzero_elements());

  // 2D spatial
  test_par->pip_set.spatial_dim = 2;
  auto test_frame2 = std::make_unique<dGFrame_spatial<2>>(test_par.get());
  test_frame2->init();
  baseline_int = 4;
  EXPECT_EQ(test_frame2->triangulation->n_active_cells(), baseline_int);
  baseline_int = 16;
  EXPECT_EQ(test_frame2->dof_handler->n_dofs(), baseline_int);

  // 3D spatial
  test_par->pip_set.spatial_dim = 3;
  auto test_frame3 = std::make_unique<dGFrame_spatial<3>>(test_par.get());
  test_frame3->init();
  baseline_int = 8;
  EXPECT_EQ(test_frame3->triangulation->n_active_cells(), baseline_int);
  baseline_int = 64;
  EXPECT_EQ(test_frame3->dof_handler->n_dofs(), baseline_int);
}

// testing spectral triangulation
TEST(dg_frames, spectral_triangulation) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();

  // general grid settings
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
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  // 1D spectral
  test_par->pip_set.spectral_dim = 1;
  auto test_frame1 = std::make_unique<dGFrame_spectral<1>>(test_par.get());
  test_frame1->init();
  baseline_int = 2;
  EXPECT_EQ(test_frame1->triangulation->n_active_cells(), baseline_int);
  baseline_int = 4;
  EXPECT_EQ(test_frame1->dof_handler->n_dofs(), baseline_int);

  // 2D spectral
  test_par->pip_set.spectral_dim = 2;
  auto test_frame2 = std::make_unique<dGFrame_spectral<2>>(test_par.get());
  test_frame2->init();
  baseline_int = 4;
  EXPECT_EQ(test_frame2->triangulation->n_active_cells(), baseline_int);
  baseline_int = 16;
  EXPECT_EQ(test_frame2->dof_handler->n_dofs(), baseline_int);

  // 3D spectral
  test_par->pip_set.spectral_dim = 3;
  auto test_frame3 = std::make_unique<dGFrame_spectral<3>>(test_par.get());
  test_frame3->init();
  baseline_int = 8;
  EXPECT_EQ(test_frame3->triangulation->n_active_cells(), baseline_int);
  baseline_int = 64;
  EXPECT_EQ(test_frame3->dof_handler->n_dofs(), baseline_int);
}

// testing frame refinement
TEST(dg_frames, refinement) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();

  // general grid settings
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
  test_par->grid_set.spectral_min_refine_lv = 2;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_frame = std::make_unique<dGFrame_spectral<3>>(test_par.get());
  test_frame->init();

  baseline_int = 4;
  EXPECT_EQ(test_frame->max_refine_lv, baseline_int);
  baseline_int = 3;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  baseline_int = 512;
  EXPECT_EQ(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 4096;
  EXPECT_EQ(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_EQ(test_frame->sparsity->n_cols(), baseline_int);

  // global refine
  test_frame->refine((const dealii::Vector<float> *)nullptr);

  // refine lv must be +1
  baseline_int = 4;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  baseline_int = 4096;
  EXPECT_EQ(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 32768;
  EXPECT_EQ(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_EQ(test_frame->sparsity->n_cols(), baseline_int);

  // adaptive refine
  dealii::Vector<float> test_err;
  test_err.reinit(4096);
  for (decltype(test_err.size()) i = 0; i < test_err.size(); ++i)
    test_err[i] = i * 0.01;

  test_frame->refine(&test_err);

  // refine lv must be +1
  baseline_int = 5;
  EXPECT_EQ(test_frame->triangulation->n_levels(), baseline_int);
  // refined objects should not equal to old ones
  baseline_int = 4096;
  EXPECT_NE(test_frame->triangulation->n_active_cells(), baseline_int);
  baseline_int = 32768;
  EXPECT_NE(test_frame->dsp->n_cols(), baseline_int);
  EXPECT_NE(test_frame->sparsity->n_cols(), baseline_int);
}
