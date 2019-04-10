// Param class collects parameters for all
// major blocks of parameters include
// - pipeline settings
// - grid settings
//
// pipeline settings host parameters related to solving PDE(s)
// grid settings host parameters related to spatial/spectral grid information

#ifndef BIFET_PARAM_H
#define BIFET_PARAM_H

#include <string>
#include <vector>

class Param {
public:
  Param() = default;
  Param(std::string);
  Param(const Param &p) = delete;
  Param(Param &&p) = delete;
  Param &operator=(const Param &p) = delete;
  Param &operator=(Param &&p) = delete;
  virtual ~Param() = default;

  // order of structures should in accordance with
  // the order appeared in copy/move constructors/semantics

  // PDE system general parameters
  struct pipeline_setting {
    std::string sys_type;
    unsigned int spatial_dim, spectral_dim;
    unsigned int spatial_pol_order, spectral_pol_order; // polynomial order
    unsigned int step_lim;                              // time step limit
    unsigned int refine_cd;   // cooldown step for adaptive refinement
    double physical_timediff; // physical time difference between two steps
    double solver_scheme;     // 2nd order precision scheme
    bool time_dependency = false;
    double evo_lim; // evolution divergence lv
    unsigned int iteration;
    double tolerance; // solver's tolerance
  } pip_set;

  // simulation grid parameters
  struct grid_setting {
    // spatial grid resolution
    unsigned int nx1, nx2, nx3;
    // dimensionless grid limits
    double x1_min, x1_max;
    double x2_min, x2_max;
    double x3_min, x3_max;
    // spectral grid resolution
    unsigned int nq1, nq2, nq3;
    // dimensionless grid limits
    double q1_min, q1_max;
    double q2_min, q2_max;
    double q3_min, q3_max;
    // controller for refinement
    bool do_spatial_refine = false, do_spectral_refine = false;
    // global/adaptive refinement level limits
    unsigned int spatial_min_refine_lv, spatial_max_refine_lv;
    unsigned int spectral_min_refine_lv, spectral_max_refine_lv;
    // refine/coarsen ratio
    double refine_ratio, coarsen_ratio;
    // refine scheme
    std::string spatial_refine_scheme, spectral_refine_scheme;
  } grid_set;
};

#endif

// END
