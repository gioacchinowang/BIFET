#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/solution_transfer.h>

#include <param.h>
#include <propagator.h>
#include <simbox.h>
#include <solution.h>
#include <system.h>
#include <timer.h>

template class Propagator<1, 1>;
template class Propagator<2, 1>;
template class Propagator<3, 1>;
template class Propagator<1, 2>;
template class Propagator<2, 2>;
template class Propagator<3, 2>;
template class Propagator<1, 3>;
template class Propagator<2, 3>;
template class Propagator<3, 3>;

template <int spa_dim, int spe_dim>
Propagator<spa_dim, spe_dim>::Propagator(const Param *par) {
  this->physical_timediff = par->pip_set.physical_timediff;
  this->step_lim = par->pip_set.step_lim;
  this->refine_cd = par->pip_set.refine_cd;
  this->evo_lim = par->pip_set.evo_lim;
  this->iteration = par->pip_set.iteration;
  this->tolerance = par->pip_set.tolerance;
  this->solver_scheme = par->pip_set.solver_scheme;
  this->time_dependency = par->pip_set.time_dependency;
  this->do_spatial_refine = par->grid_set.do_spatial_refine;
  this->do_spectral_refine = par->grid_set.do_spectral_refine;
  this->spatial_refine_scheme = par->grid_set.spatial_refine_scheme;
  this->spectral_refine_scheme = par->grid_set.spectral_refine_scheme;
  this->simbox = std::make_unique<Simbox<spa_dim, spe_dim>>(par);
  this->solution = std::make_unique<Solution<spa_dim, spe_dim>>(par);
  this->system = std::make_unique<System<spa_dim, spe_dim>>(par);
}

template <int spa_dim, int spe_dim> void Propagator<spa_dim, spe_dim>::init() {
  this->simbox->init();
  this->solution->init(this->simbox.get());
#ifdef _OPENMP
  auto omp_Mxq =
      std::make_unique<dealii::SparseMatrix<double>>(*(this->simbox->sparsity));
  auto omp_Rxq = std::make_unique<dealii::Vector<double>>(
      this->simbox->spatial_frame->dof_handler->n_dofs() *
      this->simbox->spectral_frame->dof_handler->n_dofs());
#pragma omp parallel
  {
    std::unique_ptr<System<spa_dim, spe_dim>> private_system;
    private_system.reset(this->system->clone());
    private_system->init(this->simbox.get());
#pragma omp critical
    omp_Mxq->add(1., *(private_system->Mxq));
#pragma omp critical
    omp_Rxq->add(1., *(private_system->Rxq));
  } // parallel
  this->system->Mxq.reset(omp_Mxq.release());
  this->system->Rxq.reset(omp_Rxq.release());
#else
  this->system->init(this->simbox.get());
#endif
  this->spatial_err = std::make_unique<dealii::Vector<float>>();
  this->spectral_err = std::make_unique<dealii::Vector<float>>();
  this->evo_ref = std::make_unique<std::vector<double>>();
  this->Rxq_cache = std::make_unique<dealii::Vector<double>>();
  this->eRxq_cache = std::make_unique<dealii::Vector<double>>();
}

template <int spa_dim, int spe_dim> void Propagator<spa_dim, spe_dim>::run() {
#ifdef VERBOSE
  std::cout << std::endl
            << "===========================================" << std::endl
            << "...propagator in..." << std::endl;
  unsigned int refinement_counter{0};
#endif
#ifndef NTIMING
  Timer tmr;
  tmr.tick("init");
#endif
  this->init();
#ifndef NTIMING
  tmr.tock("init");
#endif
#ifndef NTIMING
  tmr.tick("solve");
#endif
  do {
#ifdef VERBOSE
    std::cout << std::endl
              << "at refinement lv. " << refinement_counter << std::endl;
    refinement_counter++;
#endif
    if (this->time_dependency) {
      this->solve_time_step();
      this->evo_record();
      if (this->step_idx != 0 and this->step_idx % this->refine_cd == 0) {
        this->refine(this->step_time);
      } else {
        this->pseudo_refine(this->step_time);
      }
      this->evo_step();
    } else {
      this->solve_single_step();
      this->step_idx++;
      this->refine();
    }
  } while (not this->evo_break());
#ifndef NTIMING
  tmr.tock("solve");
  tmr.print();
#endif
#ifdef VERBOSE
  std::cout << std::endl
            << "...propagator out..."
            << "===========================================" << std::endl;
#endif
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::solve_single_step() {
  this->solution->pre_constraints(this->simbox.get());
  // direct solver
  auto solver_control = std::make_unique<dealii::SparseDirectUMFPACK>();
  solver_control->initialize(*(this->system->Mxq));
  solver_control->vmult(*(this->solution->Snew), *(this->system->Rxq));
  // only need to redistribute constraints to solution
  this->solution->post_constraints(this->simbox.get());
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::solve_time_step() {
  this->solution->pre_constraints(this->simbox.get());
  // system->assemble_mass_Mxq
  this->system->assemble_mass_Mxq(this->simbox.get());
  if (this->step_idx == 0) {
    // 0th step do no solving
    // only need to redistribute constraints to solution
    this->solution->post_constraints(this->simbox.get());
  } else {
    // assemble system matrix directly on mass_Mxq
    // be aware that mass_Mxq should be reinitialised at each solving step
    // which results in re-initialising System before each solving step
    // well, the only exception is step_idx=0
    this->system->mass_Mxq->add(this->physical_timediff * this->solver_scheme,
                                *(this->system->Mxq));
    // assemble system RHS directly on eRxq_cache
    // eRxq_cache should have been reinited (in evo_record) and/or refined (in
    // refine) eRxq_cache adds the normal RHS in time-dependent problem it has
    // already been reinilialised and taken operator related part in evo_record
    // notice eRxq_cache is not inside System but in Propagator
    this->eRxq_cache->add(this->physical_timediff * this->solver_scheme,
                          *(this->system->Rxq),
                          this->physical_timediff * (1. - this->solver_scheme),
                          *(this->Rxq_cache));
    // direct solver
    auto solver_control = std::make_unique<dealii::SparseDirectUMFPACK>();
    solver_control->initialize(*(this->system->mass_Mxq));
    solver_control->vmult(*(this->solution->Snew), *(this->eRxq_cache));
    // only need to redistribute constraints to solution
    this->solution->post_constraints(this->simbox.get());
  }
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::evo_record() {
  // copy new solution
  auto solution_tmp =
      std::make_unique<dealii::Vector<double>>(*(this->solution->Snew));
  // estimate relative difference
  solution_tmp->add(-1., *(this->solution->Sold));
  double new_record = solution_tmp->l2_norm();
  if (step_idx == 0) {
    this->evo_ref->push_back(1.);
  } else {
    this->evo_ref->push_back(fabs(this->evo_cache - new_record) /
                             this->evo_cache);
  }
  this->evo_cache = new_record;
#ifdef VERBOSE
  std::cout << "step: " << this->step_idx << "\t"
            << "new evo_record: " << this->evo_cache << "\t"
            << "new evo_ref: " << this->evo_ref->back() << std::endl;
#endif
  // cache system rhs and extra_rhs
  this->Rxq_cache->reinit(*(this->system->Rxq));
  *(this->Rxq_cache) = *(this->system->Rxq);
  this->eRxq_cache->reinit(*(this->Rxq_cache));
  // eRxq = ( mass_Mxq + physical_timediff*(theta-1)*Mxq ) * Snew
  // this equation takes the operator matrix as it should be in time-independent
  // problem
  if (this->step_idx == 0) {
    this->system->mass_Mxq->add(this->physical_timediff *
                                    (this->solver_scheme - 1.),
                                *(this->system->Mxq));
  } else {
    // if not at first step
    // mass_Mxq has been modified in solve_time_step
    this->system->mass_Mxq->add(-1. * this->physical_timediff,
                                *(this->system->Mxq));
  }
  this->system->mass_Mxq->vmult(*(this->eRxq_cache), *(this->solution->Snew));
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::evo_step() {
  // estimate time difference in Gyr unit
  // by default, this->physical_tiemdiff is already set in param.cc
  // accumulate this->step_time
  this->step_time += this->physical_timediff;
  // accumulate this->step_idx
  this->step_idx++;
  // copy new solution to old solution
  // must be placed after solution refinement
  *(this->solution->Sold) = *(this->solution->Snew);
}

template <int spa_dim, int spe_dim>
bool Propagator<spa_dim, spe_dim>::evo_break() {
  // evolution step limit
  if (this->step_idx > this->step_lim)
    return true;
  // absolute/relative value limit
  // if no evolution reference is recorded
  // step_idx will dominate the dicision
  else if ((not this->evo_ref->empty()) and
           (this->evo_ref->back() < this->evo_lim))
    return true;
  else
    return false;
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::pseudo_refine(const double &step_time) {
  // waiting for refinement cooldown
  // step_time = 0 means we're either in non-evolving case or in the initial
  // step, neither cases requires refinement/reinitialisation to system
  if (step_time != 0) {
    this->system->refine(this->simbox.get(), step_time);
  }
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::refine(const double &step_time) {
  // if no refinement, take the same measure as if refinement is in cooldown
  if (not this->do_spatial_refine and not this->do_spectral_refine) {
    this->pseudo_refine(step_time);
    return;
  }
  // notice that system refinement should be done every solving step
  // while grid refinement is not necessary
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  const dealii::types::global_dof_index spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  // if do refinement in single frame
  if (this->do_spatial_refine) {
    // choose refinement scheme
    // if scheme is adaptive, cache error estimation first
    if (this->spatial_refine_scheme == "adaptive_kelly") {
      // error cacher
      const unsigned int spatial_cells{
          this->simbox->spatial_frame->triangulation->n_active_cells()};
#ifdef _OPENMP
      auto omp_spatial_err =
          std::make_unique<dealii::Vector<float>>(spatial_cells);
#pragma omp parallel
      {
#else
      this->spatial_err->reinit(spatial_cells);
#endif
        auto spatial_err_tmp =
            std::make_unique<dealii::Vector<float>>(spatial_cells);
        // take every slice of solution, esitmate the summarized error-per-cell
        auto spatial_slice =
            std::make_unique<dealii::Vector<double>>(spatial_dofs);
#ifdef _OPENMP
#pragma omp for
#endif
        for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
          // cut solution slice
          for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
            (*spatial_slice)[i] = this->solution->new_el(i, j);
          }
          dealii::KellyErrorEstimator<spa_dim>::estimate(
              *(this->simbox->spatial_frame->dof_handler),
              dealii::QGauss<spa_dim - 1>(
                  this->simbox->spatial_frame->fe->degree + 1),
              std::map<dealii::types::boundary_id,
                       const dealii::Function<spa_dim, double> *>(),
              *spatial_slice, *spatial_err_tmp);
#ifdef _OPENMP
#pragma omp critical
          *(omp_spatial_err) += *spatial_err_tmp;
#else
        *(this->spatial_err) += *spatial_err_tmp;
#endif
        }
#ifdef _OPENMP
      } // parallel
      this->spatial_err.reset(omp_spatial_err.release());
#endif
#ifdef VERBOSE
      std::cout << std::endl
                << "===========================================" << std::endl
                << "spatial frame" << std::endl
                << "adaptive Kelly error estimated: " << std::endl;
      for (auto it = this->spatial_err->begin(); it != this->spatial_err->end();
           ++it)
        std::cout << *it << std::endl;
#endif
    } else if (this->spatial_refine_scheme == "adaptive_gradient") {
      // error cacher
      const unsigned int spatial_cells{
          this->simbox->spatial_frame->triangulation->n_active_cells()};
#ifdef _OPENMP
      auto omp_spatial_err =
          std::make_unique<dealii::Vector<float>>(spatial_cells);
#pragma omp parallel
      {
#else
      this->spatial_err->reinit(spatial_cells);
#endif
        auto spatial_err_tmp =
            std::make_unique<dealii::Vector<float>>(spatial_cells);
        // take every slice of solution, esitmate the summarized error-per-cell
        auto spatial_slice =
            std::make_unique<dealii::Vector<double>>(spatial_dofs);
#ifdef _OPENMP
#pragma omp for
#endif
        for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
          // cut solution slice
          for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
            (*spatial_slice)[i] = this->solution->new_el(i, j);
          }
          dealii::DerivativeApproximation::approximate_gradient(
              *(this->simbox->spatial_frame->dof_handler), *spatial_slice,
              *spatial_err_tmp);
#ifdef _OPENMP
#pragma omp critical
          *(omp_spatial_err) += *spatial_err_tmp;
#else
        *(this->spatial_err) += *spatial_err_tmp;
#endif
        }
#ifdef _OPENMP
      } // parallel
      this->spatial_err.reset(omp_spatial_err.release());
#endif
      unsigned int cell_id = 0;
      for (const auto &cell :
           this->simbox->spatial_frame->dof_handler->active_cell_iterators()) {
        (*(this->spatial_err))[cell_id++] *=
            std::pow(cell->diameter(), 1. + spa_dim / 2.0);
      }
#ifdef VERBOSE
      std::cout << std::endl
                << "===========================================" << std::endl
                << "spatial frame" << std::endl
                << "adaptive gradient error estimated: " << std::endl;
      for (auto it = this->spatial_err->begin(); it != this->spatial_err->end();
           ++it)
        std::cout << *it << std::endl;
#endif
    } else if (this->spatial_refine_scheme == "global") {
      this->spatial_err = nullptr;
    } else {
      std::cerr << "unsupported spatial refine shceme: "
                << this->spatial_refine_scheme << std::endl;
      exit(1);
    }
    // only spatial refine
    if (not this->do_spectral_refine) {
      this->evo_refine_spatial(this->simbox.get(), this->spatial_err.get(),
                               step_time);
      return;
    }
  }
  if (this->do_spectral_refine) {
    // choose refinement scheme
    // if scheme is adaptive, cache error estimation first
    if (this->spectral_refine_scheme == "adaptive_kelly") {
      // error cacher
      const unsigned int spectral_cells{
          this->simbox->spectral_frame->triangulation->n_active_cells()};
#ifdef _OPENMP
      auto omp_spectral_err =
          std::make_unique<dealii::Vector<float>>(spectral_cells);
#pragma omp parallel
      {
#else
      this->spectral_err->reinit(spectral_cells);
#endif
        auto spectral_err_tmp =
            std::make_unique<dealii::Vector<float>>(spectral_cells);
        // take every slice of solution, esitmate the summarized error-per-cell
        auto spectral_slice =
            std::make_unique<dealii::Vector<double>>(spectral_dofs);
#ifdef _OPENMP
#pragma omp for
#endif
        for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
          // cut solution slice
          for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
            (*spectral_slice)[j] = this->solution->new_el(i, j);
          }
          dealii::KellyErrorEstimator<spe_dim>::estimate(
              *(this->simbox->spectral_frame->dof_handler),
              dealii::QGauss<spe_dim - 1>(
                  this->simbox->spectral_frame->fe->degree + 1),
              std::map<dealii::types::boundary_id,
                       const dealii::Function<spe_dim, double> *>(),
              *spectral_slice, *spectral_err_tmp);
#ifdef _OPENMP
#pragma omp critical
          *(omp_spectral_err) += *spectral_err_tmp;
#else
        *(this->spectral_err) += *spectral_err_tmp;
#endif
        }
#ifdef _OPENMP
      } // parallel
      this->spectral_err.reset(omp_spectral_err.release());
#endif
#ifdef VERBOSE
      std::cout << std::endl
                << "===========================================" << std::endl
                << "spectral frame" << std::endl
                << "adaptive Kelly error estimated: " << std::endl;
      for (auto it = this->spectral_err->begin();
           it != this->spectral_err->end(); ++it)
        std::cout << *it << std::endl;
#endif
    } else if (this->spectral_refine_scheme == "adaptive_gradient") {
      // error cacher
      const unsigned int spectral_cells{
          this->simbox->spectral_frame->triangulation->n_active_cells()};
#ifdef _OPENMP
      auto omp_spectral_err =
          std::make_unique<dealii::Vector<float>>(spectral_cells);
#pragma omp parallel
      {
#else
      this->spectral_err->reinit(spectral_cells);
#endif
        auto spectral_err_tmp =
            std::make_unique<dealii::Vector<float>>(spectral_cells);
        // take every slice of solution, esitmate the summarized error-per-cell
        auto spectral_slice =
            std::make_unique<dealii::Vector<double>>(spectral_dofs);
#ifdef _OPENMP
#pragma omp for
#endif
        for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
          // cut solution slice
          for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
            (*spectral_slice)[j] = this->solution->new_el(i, j);
          }
          dealii::DerivativeApproximation::approximate_gradient(
              *(this->simbox->spectral_frame->dof_handler), *spectral_slice,
              *spectral_err_tmp);
#ifdef _OPENMP
#pragma omp critical
          *(omp_spectral_err) += *spectral_err_tmp;
#else
        *(this->spectral_err) += *spectral_err_tmp;
#endif
        }
#ifdef _OPENMP
      } // parallel
      this->spectral_err.reset(omp_spectral_err.release());
#endif
      unsigned int cell_id = 0;
      for (const auto &cell :
           this->simbox->spectral_frame->dof_handler->active_cell_iterators()) {
        (*(this->spectral_err))[cell_id++] *=
            std::pow(cell->diameter(), 1. + spe_dim / 2.0);
      }
#ifdef VERBOSE
      std::cout << std::endl
                << "===========================================" << std::endl
                << "spectral frame" << std::endl
                << "adaptive gradient error estimated: " << std::endl;
      for (auto it = this->spectral_err->begin();
           it != this->spectral_err->end(); ++it)
        std::cout << *it << std::endl;
#endif
    } else if (this->spectral_refine_scheme == "global") {
      this->spectral_err = nullptr;
    } else {
      std::cerr << "unsupported spectral refine shceme: "
                << this->spectral_refine_scheme << std::endl;
      exit(1);
    }
    // only spectral refine
    if (not this->do_spatial_refine) {
      this->evo_refine_spectral(this->simbox.get(), this->spectral_err.get(),
                                step_time);
      return;
    }
  }
  // if do refinement in both frames
  this->evo_refine(this->simbox.get(), this->spatial_err.get(),
                   this->spectral_err.get(), step_time);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::evo_refine_spatial(
    Simbox<spa_dim, spe_dim> *simbox, const dealii::Vector<float> *spatial_err,
    const double &step_time) {
  // do Solution, Rxq_cache, eRxe_cache refinement explicitly
  if (this->time_dependency) {
    // cache number of rows before refinement
    auto pre_rows = this->solution->newrows();
    // cache number of cols before refinement
    auto pre_cols = this->solution->newcols();
    // cache Snew out to temporary holder
    auto tmp_solution =
        std::make_unique<dealii::Vector<double>>(*(this->solution->Snew));
    auto tmp_Rxq = std::make_unique<dealii::Vector<double>>(*(this->Rxq_cache));
    auto tmp_eRxq =
        std::make_unique<dealii::Vector<double>>(*(this->eRxq_cache));
    // solution trans holder
    auto spa_strans = std::make_unique<dealii::SolutionTransfer<spa_dim>>(
        *(simbox->spatial_frame->dof_handler));
    // allocate spatial/spectral domain trans temporary holder
    // dof size before frame refinement
    auto solution_spa_preslot =
        std::make_unique<dealii::Vector<double>>(pre_rows);
    auto Rxq_spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
    auto eRxq_spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
    // prepare SolutionTransfer template
    spa_strans->prepare_for_coarsening_and_refinement(*solution_spa_preslot);
    // apply refinement to frames
    simbox->refine_spatial(spatial_err);
    // IMMEDIATELY AFTER SIMBOX REFINE
    // reinit new/old solutions
    // use dof after frame refinement
    this->solution->new_reshape(simbox);
    this->solution->old_reshape(simbox);
    // cache number of rows after refinement
    auto post_rows = this->solution->newrows();
    auto post_cols = this->solution->newcols();
    // number of cols after refinement shouldn't change
    assert(post_cols == pre_cols);
    // allocate spatial/spectral domain trans temporary holder
    // odf size after frame refinement
    auto solution_spa_postslot =
        std::make_unique<dealii::Vector<double>>(post_rows);
    auto Rxq_spa_postslot = std::make_unique<dealii::Vector<double>>(post_rows);
    auto eRxq_spa_postslot =
        std::make_unique<dealii::Vector<double>>(post_rows);
    this->Rxq_cache->reinit(post_cols * post_rows);
    this->eRxq_cache->reinit(post_cols * post_rows);
    // reinterpolate col by col, trans in spatial domain
    for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
      // pass a col from Snew to spa_socket
      for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
        (*solution_spa_preslot)[i] = (*tmp_solution)[i + j * pre_rows];
        (*Rxq_spa_preslot)[i] = (*tmp_Rxq)[i + j * pre_rows];
        (*eRxq_spa_preslot)[i] = (*tmp_eRxq)[i + j * pre_rows];
      }
      // conduct interpolation
      spa_strans->interpolate(*solution_spa_preslot, *solution_spa_postslot);
      spa_strans->interpolate(*Rxq_spa_preslot, *Rxq_spa_postslot);
      spa_strans->interpolate(*eRxq_spa_preslot, *eRxq_spa_postslot);
      // apply hanging node constraints before
      simbox->spatial_frame->constraints->distribute(*solution_spa_postslot);
      simbox->spatial_frame->constraints->distribute(*Rxq_spa_postslot);
      simbox->spatial_frame->constraints->distribute(*eRxq_spa_postslot);
      // passing postslot into Snew
      for (decltype(post_rows) i = 0; i < post_rows; ++i) {
        (*(this->solution->Snew))[i + j * post_rows] =
            (*solution_spa_postslot)[i];
        (*(this->Rxq_cache))[i + j * post_rows] = (*Rxq_spa_postslot)[i];
        (*(this->eRxq_cache))[i + j * post_rows] = (*eRxq_spa_postslot)[i];
      }
    }
  } else {
    this->solution->refine_spatial(simbox, spatial_err);
  }
  this->system->refine(simbox, step_time);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::evo_refine_spectral(
    Simbox<spa_dim, spe_dim> *simbox, const dealii::Vector<float> *spectral_err,
    const double &step_time) {
  // do Solution, Rxq_cache, eRxe_cache refinement explicitly
  if (this->time_dependency) {
    // cache number of rows before refinement
    auto pre_rows = this->solution->newrows();
    // cache number of cols before refinement
    auto pre_cols = this->solution->newcols();
    // cache Snew out to temporary holder
    auto tmp_solution =
        std::make_unique<dealii::Vector<double>>(*(this->solution->Snew));
    auto tmp_Rxq = std::make_unique<dealii::Vector<double>>(*(this->Rxq_cache));
    auto tmp_eRxq =
        std::make_unique<dealii::Vector<double>>(*(this->eRxq_cache));
    // solution trans holder
    auto spe_strans = std::make_unique<dealii::SolutionTransfer<spe_dim>>(
        *(simbox->spectral_frame->dof_handler));
    // allocate spatial/spectral domain trans temporary holder
    // dof size before frame refinement
    auto solution_spe_preslot =
        std::make_unique<dealii::Vector<double>>(pre_cols);
    auto Rxq_spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
    auto eRxq_spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
    // prepare SolutionTransfer template
    spe_strans->prepare_for_coarsening_and_refinement(*solution_spe_preslot);
    // apply refinement to frames
    simbox->refine_spectral(spectral_err);
    // IMMEDIATELY AFTER SIMBOX REFINE
    // reinit new/old solutions
    // use dof after frame refinement
    this->solution->new_reshape(simbox);
    this->solution->old_reshape(simbox);
    // cache number of cols after refinement
    auto post_cols = this->solution->newcols();
    auto post_rows = this->solution->newrows();
    // number of rows after refinement shouldn't change
    assert(post_rows == pre_rows);
    // allocate spatial/spectral domain trans temporary holder
    // odf size after frame refinement
    auto solution_spe_postslot =
        std::make_unique<dealii::Vector<double>>(post_cols);
    auto Rxq_spe_postslot = std::make_unique<dealii::Vector<double>>(post_cols);
    auto eRxq_spe_postslot =
        std::make_unique<dealii::Vector<double>>(post_cols);
    this->Rxq_cache->reinit(post_cols * post_rows);
    this->eRxq_cache->reinit(post_cols * post_rows);
    // reinterpolate row by row, trans in spectral domain
    for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
      // pass a row from tmp_solution to spe_socket
      for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
        (*solution_spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
        (*Rxq_spe_preslot)[j] = (*tmp_Rxq)[i + j * pre_rows];
        (*eRxq_spe_preslot)[j] = (*tmp_eRxq)[i + j * pre_rows];
      }
      // conduct interpolation
      spe_strans->interpolate(*solution_spe_preslot, *solution_spe_postslot);
      spe_strans->interpolate(*Rxq_spe_preslot, *Rxq_spe_postslot);
      spe_strans->interpolate(*eRxq_spe_preslot, *eRxq_spe_postslot);
      // apply hanging node constraints before
      simbox->spectral_frame->constraints->distribute(*solution_spe_postslot);
      simbox->spectral_frame->constraints->distribute(*Rxq_spe_postslot);
      simbox->spectral_frame->constraints->distribute(*eRxq_spe_postslot);
      // passing postslot into reinitiated Snew
      for (decltype(post_cols) j = 0; j < post_cols; ++j) {
        (*(this->solution->Snew))[i + j * pre_rows] =
            (*solution_spe_postslot)[j];
        (*(this->Rxq_cache))[i + j * post_rows] = (*Rxq_spe_postslot)[j];
        (*(this->eRxq_cache))[i + j * post_rows] = (*eRxq_spe_postslot)[j];
      }
    }
  } else {
    this->solution->refine_spectral(simbox, spectral_err);
  }
  this->system->refine(simbox, step_time);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::evo_refine(
    Simbox<spa_dim, spe_dim> *simbox, const dealii::Vector<float> *spatial_err,
    const dealii::Vector<float> *spectral_err, const double &step_time) {
  // do Solution, Rxq_cache, eRxe_cache refinement explicitly
  if (this->time_dependency) {
    // cache number of rows before refinement
    auto pre_rows = this->solution->newrows();
    // cache number of cols before refinement
    auto pre_cols = this->solution->newcols();
    // cache Snew out to temporary holder
    auto tmp_solution =
        std::make_unique<dealii::Vector<double>>(*(this->solution->Snew));
    auto tmp_Rxq = std::make_unique<dealii::Vector<double>>(*(this->Rxq_cache));
    auto tmp_eRxq =
        std::make_unique<dealii::Vector<double>>(*(this->eRxq_cache));
    // solution trans holder
    auto spa_strans = std::make_unique<dealii::SolutionTransfer<spa_dim>>(
        *(simbox->spatial_frame->dof_handler));
    auto spe_strans = std::make_unique<dealii::SolutionTransfer<spe_dim>>(
        *(simbox->spectral_frame->dof_handler));
    // allocate spatial/spectral domain trans temporary holder
    // dof size before frame refinement
    auto solution_spa_preslot =
        std::make_unique<dealii::Vector<double>>(pre_rows);
    auto Rxq_spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
    auto eRxq_spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
    auto solution_spe_preslot =
        std::make_unique<dealii::Vector<double>>(pre_cols);
    auto Rxq_spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
    auto eRxq_spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
    // prepare SolutionTransfer template
    spa_strans->prepare_for_coarsening_and_refinement(*solution_spa_preslot);
    spe_strans->prepare_for_coarsening_and_refinement(*solution_spe_preslot);
    // apply refinement to frames
    simbox->refine(spatial_err, spectral_err);
    // IMMEDIATELY AFTER SIMBOX REFINE
    // reinit new/old solutions
    // use dof after frame refinement
    this->solution->new_reshape(simbox);
    this->solution->old_reshape(simbox);
    // cache number of rows after refinement
    auto post_rows = this->solution->newrows();
    // cache number of cols after refinement
    auto post_cols = this->solution->newcols();
    // allocate spatial/spectral domain trans temporary holder
    // odf size after frame refinement
    auto solution_spa_postslot =
        std::make_unique<dealii::Vector<double>>(post_rows);
    auto Rxq_spa_postslot = std::make_unique<dealii::Vector<double>>(post_rows);
    auto eRxq_spa_postslot =
        std::make_unique<dealii::Vector<double>>(post_rows);
    auto solution_spe_postslot =
        std::make_unique<dealii::Vector<double>>(post_cols);
    auto Rxq_spe_postslot = std::make_unique<dealii::Vector<double>>(post_cols);
    auto eRxq_spe_postslot =
        std::make_unique<dealii::Vector<double>>(post_cols);
    this->Rxq_cache->reinit(post_cols * post_rows);
    this->eRxq_cache->reinit(post_cols * post_rows);
    // if post_rows < pre_rows, refined Snew vector cannot hold enough info
    // after spectral domain trans except using another cache holder, the
    // algorithm of transform is the same in two cases
    if (pre_rows > post_rows) {
      // intermediate info holder
      auto mid_solution =
          std::make_unique<dealii::Vector<double>>(pre_rows * post_cols);
      auto mid_Rxq =
          std::make_unique<dealii::Vector<double>>(pre_rows * post_cols);
      auto mid_eRxq =
          std::make_unique<dealii::Vector<double>>(pre_rows * post_cols);
      // reinterpolate row by row, trans in spectral domain
      for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
        // pass a row from tmp_solution to spe_socket
        for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
          (*solution_spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
          (*Rxq_spe_preslot)[j] = (*tmp_Rxq)[i + j * pre_rows];
          (*eRxq_spe_preslot)[j] = (*tmp_eRxq)[i + j * pre_rows];
        }
        // conduct interpolation
        spe_strans->interpolate(*solution_spe_preslot, *solution_spe_postslot);
        spe_strans->interpolate(*Rxq_spe_preslot, *Rxq_spe_postslot);
        spe_strans->interpolate(*eRxq_spe_preslot, *eRxq_spe_postslot);
        // apply hanging node constraints before
        // passing postslot into mid_solution
        simbox->spectral_frame->constraints->distribute(*solution_spe_postslot);
        simbox->spectral_frame->constraints->distribute(*Rxq_spe_postslot);
        simbox->spectral_frame->constraints->distribute(*eRxq_spe_postslot);
        // refined Snew has not enough row lines to hold
        for (decltype(post_cols) j = 0; j < post_cols; ++j) {
          (*mid_solution)[i + j * pre_rows] = (*solution_spe_postslot)[j];
          (*mid_Rxq)[i + j * post_rows] = (*Rxq_spe_postslot)[j];
          (*mid_eRxq)[i + j * post_rows] = (*eRxq_spe_postslot)[j];
        }
      }
      // reinterpolate col by col, trans in spatial domain
      for (decltype(post_cols) j = 0; j < post_cols; ++j) {
        // pass a col from Snew to spa_socket
        for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
          (*solution_spa_preslot)[i] = (*mid_solution)[i + j * pre_rows];
          (*Rxq_spa_preslot)[i] = (*mid_Rxq)[i + j * pre_rows];
          (*eRxq_spa_preslot)[i] = (*mid_eRxq)[i + j * pre_rows];
        }
        // conduct interpolation
        spa_strans->interpolate(*solution_spa_preslot, *solution_spa_postslot);
        spa_strans->interpolate(*Rxq_spa_preslot, *Rxq_spa_postslot);
        spa_strans->interpolate(*eRxq_spa_preslot, *eRxq_spa_postslot);
        // apply hanging node constraints before
        simbox->spatial_frame->constraints->distribute(*solution_spa_postslot);
        simbox->spatial_frame->constraints->distribute(*Rxq_spa_postslot);
        simbox->spatial_frame->constraints->distribute(*eRxq_spa_postslot);
        // passing postslot into Snew
        for (decltype(post_rows) i = 0; i < post_rows; ++i) {
          (*(this->solution->Snew))[i + j * post_rows] =
              (*solution_spa_postslot)[i];
          (*(this->Rxq_cache))[i + j * post_rows] = (*Rxq_spa_postslot)[i];
          (*(this->eRxq_cache))[i + j * post_rows] = (*eRxq_spa_postslot)[i];
        }
      }
    }
    // if Snew is large enough to hold intermediate info
    else {
      // reinterpolate row by row, trans in spectral domain
      for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
        // pass a row from tmp_solution to spe_socket
        for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
          (*solution_spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
          (*Rxq_spe_preslot)[j] = (*tmp_Rxq)[i + j * pre_rows];
          (*eRxq_spe_preslot)[j] = (*tmp_eRxq)[i + j * pre_rows];
        }
        // conduct interpolation
        spe_strans->interpolate(*solution_spe_preslot, *solution_spe_postslot);
        spe_strans->interpolate(*Rxq_spe_preslot, *Rxq_spe_postslot);
        spe_strans->interpolate(*eRxq_spe_preslot, *eRxq_spe_postslot);
        // apply hanging node constraints before
        simbox->spectral_frame->constraints->distribute(*solution_spe_postslot);
        simbox->spectral_frame->constraints->distribute(*Rxq_spe_postslot);
        simbox->spectral_frame->constraints->distribute(*eRxq_spe_postslot);
        // passing postslot into reinitiated Snew
        for (decltype(post_cols) j = 0; j < post_cols; ++j) {
          (*(this->solution->Snew))[i + j * post_rows] =
              (*solution_spe_postslot)[j];
          (*(this->Rxq_cache))[i + j * post_rows] = (*Rxq_spe_postslot)[j];
          (*(this->eRxq_cache))[i + j * post_rows] = (*eRxq_spe_postslot)[j];
        }
      }
      // reinterpolate col by col, trans in spatial domain
      for (decltype(post_cols) j = 0; j < post_cols; ++j) {
        // pass a col from Snew to spa_socket
        for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
          (*solution_spa_preslot)[i] =
              (*(this->solution->Snew))[i + j * post_rows];
          (*Rxq_spa_preslot)[i] = (*(this->Rxq_cache))[i + j * pre_rows];
          (*eRxq_spa_preslot)[i] = (*(this->eRxq_cache))[i + j * pre_rows];
        }
        // conduct interpolation
        spa_strans->interpolate(*solution_spa_preslot, *solution_spa_postslot);
        spa_strans->interpolate(*Rxq_spa_preslot, *Rxq_spa_postslot);
        spa_strans->interpolate(*eRxq_spa_preslot, *eRxq_spa_postslot);
        // apply hanging node constraints before
        simbox->spatial_frame->constraints->distribute(*solution_spa_postslot);
        simbox->spatial_frame->constraints->distribute(*Rxq_spa_postslot);
        simbox->spatial_frame->constraints->distribute(*eRxq_spa_postslot);
        // pass postslot into Snew
        for (decltype(post_rows) i = 0; i < post_rows; ++i) {
          (*(this->solution->Snew))[i + j * post_rows] =
              (*solution_spa_postslot)[i];
          (*(this->Rxq_cache))[i + j * post_rows] = (*Rxq_spa_postslot)[i];
          (*(this->eRxq_cache))[i + j * post_rows] = (*eRxq_spa_postslot)[i];
        }
      }
    }
  } else {
    this->solution->refine(simbox, spatial_err, spectral_err);
  }
  this->system->refine(simbox, step_time);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::density_snapshot(const std::string header) {
  // integrate solution over spectral domain
  // (reinit density field)
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  this->density_field = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  // preparation for spectral integration
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      this->simbox->spectral_frame->fe->degree + 1);
  // (fe_values in spectral domain)
  auto spectral_fev = std::make_unique<dealii::FEValues<spe_dim>>(
      *(this->simbox->spectral_frame->fe), *spectral_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // (spectral domain dofs per cell)
  const dealii::types::global_dof_index spectral_dpc =
      spectral_fev->dofs_per_cell;
  // (num of spectral quadrature points per cell)
  const dealii::types::global_dof_index spectral_q_points =
      spectral_quadrature_formula->size();
  // (spectral local 2 global index mapping)
  auto spectral_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);

  // (loop through global spatial dof index)
  for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
    // (loop through spectral cells)
    for (const auto &spectral_cell :
         this->simbox->spectral_frame->dof_handler->active_cell_iterators()) {
      spectral_fev->reinit(spectral_cell);
      spectral_cell->get_dof_indices(*spectral_l2g);
      for (dealii::types::global_dof_index spectral_qid = 0;
           spectral_qid < spectral_q_points; ++spectral_qid) {
        for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
             ++alpha) {
          (*(this->density_field))[i] +=
              this->solution->new_el(i, spectral_l2g->at(alpha)) *
              spectral_fev->shape_value(alpha, spectral_qid) *
              spectral_fev->JxW(spectral_qid);
        } // alpha
      }   // spectral q
    }     // spectral cell
  }       // spatial dof
  // output normally
  auto data_out = std::make_unique<dealii::DataOut<spa_dim>>();
  data_out->attach_dof_handler(*(this->simbox->spatial_frame->dof_handler));
  data_out->add_data_vector(*(this->density_field), "U");
  data_out->build_patches();
  const std::string filename =
      header + "_density_" +
      dealii::Utilities::int_to_string(this->step_idx, 4) + ".vtk";
  std::ofstream output(filename);
  data_out->write_vtk(output);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::spectral_snapshot(
    const dealii::Point<spa_dim, double> &x0, const std::string header) {
  const dealii::types::global_dof_index spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spe_slice = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(this->simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto x_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spatial_frame->dof_handler), x0);
  field_spatial->set_active_cell(x_id);
  for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
    for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i)
      (*tmp_spatial)[i] = this->solution->new_el(i, j);
    (*spe_slice)[j] = field_spatial->value(x0);
  }
  // output normally
  auto data_out = std::make_unique<dealii::DataOut<spe_dim>>();
  data_out->attach_dof_handler(*(this->simbox->spectral_frame->dof_handler));
  data_out->add_data_vector(*(spe_slice), "U");
  data_out->build_patches();
  const std::string filename =
      header + "_spec_snapshot_" +
      dealii::Utilities::int_to_string(this->step_idx, 4) + ".vtk";
  std::ofstream output(filename);
  data_out->write_vtk(output);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::spatial_snapshot(
    const dealii::Point<spe_dim, double> &q0, const std::string header) {
  const dealii::types::global_dof_index spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spa_slice = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto tmp_spectral = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(this->simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto q_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spectral_frame->dof_handler), q0);
  field_spectral->set_active_cell(q_id);
  for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
    for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j)
      (*tmp_spectral)[j] = this->solution->new_el(i, j);
    (*spa_slice)[i] = field_spectral->value(q0);
  }
  // output normally
  auto data_out = std::make_unique<dealii::DataOut<spa_dim>>();
  data_out->attach_dof_handler(*(this->simbox->spatial_frame->dof_handler));
  data_out->add_data_vector(*(spa_slice), "U");
  data_out->build_patches();
  const std::string filename =
      header + "_spat_snapshot_" +
      dealii::Utilities::int_to_string(this->step_idx, 4) + ".vtk";
  std::ofstream output(filename);
  data_out->write_vtk(output);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::Rxq_spectral_snapshot(
    const dealii::Point<spa_dim, double> &x0, const std::string header) {
  const dealii::types::global_dof_index spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spe_slice = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(this->simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto x_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spatial_frame->dof_handler), x0);
  field_spatial->set_active_cell(x_id);
  for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j) {
    for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i)
      (*tmp_spatial)[i] = (*this->system->Rxq)[j * spatial_dofs + i];
    (*spe_slice)[j] = field_spatial->value(x0);
  }
  // output normally
  auto data_out = std::make_unique<dealii::DataOut<spe_dim>>();
  data_out->attach_dof_handler(*(this->simbox->spectral_frame->dof_handler));
  data_out->add_data_vector(*(spe_slice), "U");
  data_out->build_patches();
  const std::string filename =
      header + "_spec_snapshot_" +
      dealii::Utilities::int_to_string(this->step_idx, 4) + ".vtk";
  std::ofstream output(filename);
  data_out->write_vtk(output);
}

template <int spa_dim, int spe_dim>
void Propagator<spa_dim, spe_dim>::Rxq_spatial_snapshot(
    const dealii::Point<spe_dim, double> &q0, const std::string header) {
  const dealii::types::global_dof_index spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const dealii::types::global_dof_index spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spa_slice = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto tmp_spectral = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(this->simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto q_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spectral_frame->dof_handler), q0);
  field_spectral->set_active_cell(q_id);
  for (dealii::types::global_dof_index i = 0; i < spatial_dofs; ++i) {
    for (dealii::types::global_dof_index j = 0; j < spectral_dofs; ++j)
      (*tmp_spectral)[j] = (*this->system->Rxq)[j * spatial_dofs + i];
    (*spa_slice)[i] = field_spectral->value(q0);
  }
  // output normally
  auto data_out = std::make_unique<dealii::DataOut<spa_dim>>();
  data_out->attach_dof_handler(*(this->simbox->spatial_frame->dof_handler));
  data_out->add_data_vector(*(spa_slice), "U");
  data_out->build_patches();
  const std::string filename =
      header + "_spat_snapshot_" +
      dealii::Utilities::int_to_string(this->step_idx, 4) + ".vtk";
  std::ofstream output(filename);
  data_out->write_vtk(output);
}

template <int spa_dim, int spe_dim>
double Propagator<spa_dim, spe_dim>::solution_dist(
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  return this->solution->evaluate(this->simbox.get(), x0, q0);
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spa_dim, double>
Propagator<spa_dim, spe_dim>::solution_distdx(
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  return this->solution->evaluatedx(this->simbox.get(), x0, q0);
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double>
Propagator<spa_dim, spe_dim>::solution_distdq(
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  return this->solution->evaluatedq(this->simbox.get(), x0, q0);
}

// END
