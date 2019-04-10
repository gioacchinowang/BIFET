#include <cstdlib>
#include <iostream>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <deal.II/base/point.h>

#include <param.h>
#include <propagator.h>
#include <timer.h>

// this routine is designed for performance profiling
int main(int /*argc*/, char ** /*argv*/) {
#if !defined(NTIMING) && !defined(NDEBUG)
  try {
#ifdef _OPENMP
    std::cout << omp_get_max_threads() << " threads avail" << std::endl;
#endif
    for (int N = 40; N <= 200; N += 40) {
      Timer tmr;
      // tmr.tick("main");
      auto profile_par = std::make_unique<Param>();
      profile_par->pip_set.iteration = 1000;
      profile_par->pip_set.tolerance = 1.e-12;
      // spatial
      profile_par->pip_set.spatial_pol_order = 1;
      profile_par->grid_set.x1_max = 1.;
      profile_par->grid_set.x1_min = 0.;
      profile_par->grid_set.x2_max = 1.;
      profile_par->grid_set.x2_min = 0.;
      profile_par->grid_set.x3_max = 1.;
      profile_par->grid_set.x3_min = 0.;
      profile_par->grid_set.nx1 = N;
      profile_par->grid_set.nx2 = N;
      profile_par->grid_set.nx3 = N;
      profile_par->pip_set.spatial_dim = 1;
      // spectral
      profile_par->pip_set.spectral_pol_order = 1;
      profile_par->grid_set.q1_max = 1;
      profile_par->grid_set.q1_min = -1;
      profile_par->grid_set.q2_max = 1;
      profile_par->grid_set.q2_min = -1;
      profile_par->grid_set.q3_max = 1;
      profile_par->grid_set.q3_min = -1;
      profile_par->grid_set.nq1 = N;
      profile_par->grid_set.nq2 = N;
      profile_par->grid_set.nq3 = N;
      profile_par->pip_set.spectral_dim = 1;
      // grid refine limits
      profile_par->grid_set.spatial_min_refine_lv = 1;
      profile_par->grid_set.spatial_max_refine_lv = 2;
      profile_par->grid_set.spectral_min_refine_lv = 1;
      profile_par->grid_set.spectral_max_refine_lv = 2;
      profile_par->grid_set.refine_ratio = 0.5;
      profile_par->grid_set.coarsen_ratio = 0;
      profile_par->grid_set.do_spatial_refine = true;
      profile_par->grid_set.do_spectral_refine = true;
      profile_par->grid_set.spatial_refine_scheme = "adaptive_kelly";
      profile_par->grid_set.spectral_refine_scheme = "adaptive_gradient";

      // build propagator
      auto profile_prop = std::make_unique<Propagator<3, 3>>(profile_par.get());

      tmr.tick("init");
      profile_prop->init();
      tmr.tock("init");

      std::cout
          << "domain dofs: "
          << profile_prop->simbox->spatial_frame->dof_handler->n_dofs() *
                 profile_prop->simbox->spectral_frame->dof_handler->n_dofs()
          << std::endl;

      tmr.tick("solver");
      profile_prop->solve_single_step();
      tmr.tock("solver");

      tmr.tick("refine");
      profile_prop->refine();
      tmr.tock("refine");

      // tmr.tick("output_density");
      // profile_prop->output_density("default");
      // tmr.tock("output_density");
      // tmr.tock("main");

      tmr.print();
    }
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
#endif
  return EXIT_SUCCESS;
}
