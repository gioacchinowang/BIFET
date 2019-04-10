// unit tests for propagator class
// solver tests are handled in examples
// feel free to add more

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <deal.II/base/point.h>

#include <param.h>
#include <propagator.h>
#include <solution.h>

// test propagator allocation
TEST(propagator, init) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();
  // 1D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->pip_set.spatial_dim = 1;
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->pip_set.spectral_dim = 1;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 3;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spectral_min_refine_lv = 3;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.refine_ratio = 0.3;
  test_par->grid_set.coarsen_ratio = 0;

  auto test_prop = std::make_unique<Propagator<1, 1>>(test_par.get());
  test_prop->init();

  // simbox allocated
  baseline_int = 17;
  EXPECT_EQ(test_prop->simbox->spatial_frame->dof_handler->n_dofs(),
            baseline_int);
  // solution allocated
  EXPECT_EQ(test_prop->solution->n_rows_new, baseline_int);
  // system allocated
  EXPECT_EQ(test_prop->system->Rxq->size(), baseline_int * baseline_int);

  test_prop = std::make_unique<Propagator<1, 1>>(test_par.get());
  test_prop->init();

  // simbox allocated
  baseline_int = 17;
  EXPECT_EQ(test_prop->simbox->spatial_frame->dof_handler->n_dofs(),
            baseline_int);
  // solution allocated
  EXPECT_EQ(test_prop->solution->n_rows_new, baseline_int);
  // system allocated
  EXPECT_EQ(test_prop->system->Rxq->size(), baseline_int * baseline_int);
}

// test refine function
TEST(propagator, refine) {
  unsigned int baseline_int;
  auto test_par = std::make_unique<Param>();
  // 1D spatial
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.nx1 = 3;
  test_par->pip_set.spatial_dim = 1;
  // 1D spectral
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 0;
  test_par->grid_set.q1_min = -2;
  test_par->grid_set.nq1 = 3;
  test_par->pip_set.spectral_dim = 1;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 1;
  test_par->grid_set.spatial_max_refine_lv = 3;
  test_par->grid_set.spectral_min_refine_lv = 1;
  test_par->grid_set.spectral_max_refine_lv = 3;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = true;

  // build propagator
  auto test_crudeprop = std::make_unique<Propagator<1, 1>>(test_par.get());
  test_crudeprop->init();

  // run refine routine
  test_par->grid_set.spatial_refine_scheme = "global";
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_fineprop1 = std::make_unique<Propagator<1, 1>>(test_par.get());
  test_fineprop1->init();
  test_fineprop1->refine();

  // triangulation lv +1
  baseline_int = 3;
  EXPECT_EQ(test_fineprop1->simbox->spatial_frame->triangulation->n_levels(),
            baseline_int);
  EXPECT_EQ(test_fineprop1->simbox->spectral_frame->triangulation->n_levels(),
            baseline_int);
}

// END
