#include <cassert>
#include <memory>

#include <deal.II/base/point.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <namespace_toolkit.h>
#include <simbox.h>
#include <solution.h>

template class Solution<1, 1>;
template class Solution<2, 1>;
template class Solution<3, 1>;
template class Solution<1, 2>;
template class Solution<2, 2>;
template class Solution<3, 2>;
template class Solution<1, 3>;
template class Solution<2, 3>;
template class Solution<3, 3>;

template <int spa_dim, int spe_dim> Solution<spa_dim, spe_dim>::Solution() {
  this->Snew = std::make_unique<dealii::Vector<double>>();
  this->Sold = std::make_unique<dealii::Vector<double>>();
  this->spatial =
      std::make_unique<typename Solution<spa_dim, spe_dim>::Spatial_initial>();
  this->spectral =
      std::make_unique<typename Solution<spa_dim, spe_dim>::Spectral_initial>();
}

template <int spa_dim, int spe_dim>
Solution<spa_dim, spe_dim>::Solution(const Param *) {
  this->Snew = std::make_unique<dealii::Vector<double>>();
  this->Sold = std::make_unique<dealii::Vector<double>>();
  this->spatial =
      std::make_unique<typename Solution<spa_dim, spe_dim>::Spatial_initial>();
  this->spectral =
      std::make_unique<typename Solution<spa_dim, spe_dim>::Spectral_initial>();
}

template <int spa_dim, int spe_dim>
Solution<spa_dim, spe_dim>::Solution(const Solution<spa_dim, spe_dim> &s) {
  this->Snew = std::make_unique<dealii::Vector<double>>(*(s.Snew));
  this->Sold = std::make_unique<dealii::Vector<double>>(*(s.Sold));
  this->spatial.reset(s.spatial->clone());
  this->spectral.reset(s.spectral->clone());
}

template <int spa_dim, int spe_dim>
Solution<spa_dim, spe_dim>::Solution(Solution<spa_dim, spe_dim> &&s) {
  this->Snew.reset(std::move(s.Snew.release()));
  this->Sold.reset(std::move(s.Sold.release()));
  this->spatial.reset(std::move(s.spatial.release()));
  this->spectral.reset(std::move(s.spectral.release()));
}

template <int spa_dim, int spe_dim>
Solution<spa_dim, spe_dim> &Solution<spa_dim, spe_dim>::
operator=(const Solution<spa_dim, spe_dim> &s) noexcept {
  *(this->Snew) = *(s.Snew);
  *(this->Sold) = *(s.Sold);
  *(this->spatial) = *(s.spatial);
  *(this->spectral) = *(s.spectral);
  return *this;
}

template <int spa_dim, int spe_dim>
Solution<spa_dim, spe_dim> &Solution<spa_dim, spe_dim>::
operator=(Solution<spa_dim, spe_dim> &&s) noexcept {
  *(this->Snew) = std::move(*(s.Snew));
  *(this->Sold) = std::move(*(s.Sold));
  *(this->spatial) = std::move(*(s.spatial));
  *(this->spectral) = std::move(*(s.spectral));
  return *this;
}

template <int spa_dim, int spe_dim>
double Solution<spa_dim, spe_dim>::new_el(const unsigned int &row_idx,
                                          const unsigned int &col_idx) const {
  // in Fortran style
  return (*(this->Snew))[col_idx * this->n_rows_new + row_idx];
}

template <int spa_dim, int spe_dim>
double &Solution<spa_dim, spe_dim>::new_el(const unsigned int &row_idx,
                                           const unsigned int &col_idx) {
  // in Fortran style
  return (*(this->Snew))[col_idx * this->n_rows_new + row_idx];
}

template <int spa_dim, int spe_dim>
double Solution<spa_dim, spe_dim>::old_el(const unsigned int &row_idx,
                                          const unsigned int &col_idx) const {
  // in Fortran style
  return (*(this->Sold))[col_idx * this->n_rows_old + row_idx];
}

template <int spa_dim, int spe_dim>
double &Solution<spa_dim, spe_dim>::old_el(const unsigned int &row_idx,
                                           const unsigned int &col_idx) {
  // in Fortran style
  return (*(this->Sold))[col_idx * this->n_rows_old + row_idx];
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::new_reshape(
    const Simbox<spa_dim, spe_dim> *simbox) {
  this->n_rows_new = simbox->spatial_frame->dof_handler->n_dofs();
  this->n_cols_new = simbox->spectral_frame->dof_handler->n_dofs();
  this->Snew->reinit(this->n_rows_new * this->n_cols_new);
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::old_reshape(
    const Simbox<spa_dim, spe_dim> *simbox) {
  this->n_rows_old = simbox->spatial_frame->dof_handler->n_dofs();
  this->n_cols_old = simbox->spectral_frame->dof_handler->n_dofs();
  this->Sold->reinit(this->n_rows_old * this->n_cols_old);
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::init(const Simbox<spa_dim, spe_dim> *simbox) {
  this->new_reshape(simbox);
  this->old_reshape(simbox);
  // interpolate initial condition
  this->apply_initial_condition(simbox);
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::apply_initial_condition(
    const Simbox<spa_dim, spe_dim> *simbox) {
  // temporary vectors for spatial/spectral domain interpolation
  auto spe_slice = std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto spa_slice = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  // row direction (spectral domain) interpolation
  dealii::VectorTools::interpolate(*(simbox->spectral_frame->dof_handler),
                                   *(this->spectral), *(spe_slice));
  // column direction (spatial domain) interpolation
  dealii::VectorTools::interpolate(*(simbox->spatial_frame->dof_handler),
                                   *(this->spatial), *(spa_slice));
  // cross produce of two vectors into solution vector
  toolkit::Kronecker_product<dealii::Vector<double>>(
      spe_slice.get(), spa_slice.get(), this->Snew.get());
}

template <int spa_dim, int spe_dim>
double Solution<spa_dim, spe_dim>::evaluate(
    const Simbox<spa_dim, spe_dim> *simbox,
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  auto tmp_spectral =
      std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto c_id = dealii::GridTools::find_active_cell_around_point(
      *(simbox->spatial_frame->dof_handler), x0);
  field_spatial->set_active_cell(c_id);
  // loop through "cols" (dof in spectral domain) of Snew
  for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j) {
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i) {
      // Snew memory allocated in Fortran style
      (*tmp_spatial)[i] = this->new_el(i, j);
    }
    (*tmp_spectral)[j] = field_spatial->value(x0);
  }
  return field_spectral->value(q0);
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spa_dim, double> Solution<spa_dim, spe_dim>::evaluatedx(
    const Simbox<spa_dim, spe_dim> *simbox,
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  auto tmp_spectral =
      std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto q_id = dealii::GridTools::find_active_cell_around_point(
      *(simbox->spectral_frame->dof_handler), q0);
  field_spectral->set_active_cell(q_id);
  for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i) {
    for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j) {
      (*tmp_spectral)[j] = this->new_el(i, j);
    }
    (*tmp_spatial)[i] = field_spectral->value(q0);
  }
  return field_spatial->gradient(x0);
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Solution<spa_dim, spe_dim>::evaluatedq(
    const Simbox<spa_dim, spe_dim> *simbox,
    const dealii::Point<spa_dim, double> &x0,
    const dealii::Point<spe_dim, double> &q0) const {
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  auto tmp_spectral =
      std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto x_id = dealii::GridTools::find_active_cell_around_point(
      *(simbox->spatial_frame->dof_handler), x0);
  field_spatial->set_active_cell(x_id);
  // loop through "cols" (dof in spectral domain) of Snew
  for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j) {
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i) {
      // Snew memory allocated in Fortran style
      (*tmp_spatial)[i] = this->new_el(i, j);
    }
    (*tmp_spectral)[j] = field_spatial->value(x0);
  }
  return field_spectral->gradient(q0);
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::refine(
    Simbox<spa_dim, spe_dim> *simbox, const dealii::Vector<float> *spatial_err,
    const dealii::Vector<float> *spectral_err) {
  // cache number of rows before refinement
  auto pre_rows = this->n_rows_new;
  // cache number of cols before refinement
  auto pre_cols = this->n_cols_new;
  // cache Snew out to temporary holder
  auto tmp_solution = std::make_unique<dealii::Vector<double>>(*(this->Snew));
  // solution trans holder
  auto spa_strans = std::make_unique<dealii::SolutionTransfer<spa_dim>>(
      *(simbox->spatial_frame->dof_handler));
  auto spe_strans = std::make_unique<dealii::SolutionTransfer<spe_dim>>(
      *(simbox->spectral_frame->dof_handler));
  // allocate spatial/spectral domain trans temporary holder
  // dof size before frame refinement
  auto spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
  auto spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
  // prepare SolutionTransfer template
  spe_strans->prepare_for_coarsening_and_refinement(*spe_preslot);
  spa_strans->prepare_for_coarsening_and_refinement(*spa_preslot);
  // apply refinement to frames
  simbox->refine(spatial_err, spectral_err);
  // IMMEDIATELY AFTER SIMBOX REFINE
  // reinit new/old solutions
  // use dof after frame refinement
  this->new_reshape(simbox);
  this->old_reshape(simbox);
  // cache number of rows after refinement
  auto post_rows = this->n_rows_new;
  // cache number of cols after refinement
  auto post_cols = this->n_cols_new;
  // allocate spatial/spectral domain trans temporary holder
  // odf size after frame refinement
  auto spe_postslot = std::make_unique<dealii::Vector<double>>(post_cols);
  auto spa_postslot = std::make_unique<dealii::Vector<double>>(post_rows);
  // if post_rows < pre_rows, refined Snew vector cannot hold enough info after
  // spectral domain trans except using another cache holder, the algorithm of
  // transform is the same in two cases
  if (pre_rows > post_rows) {
    // intermediate info holder
    auto mid_solution =
        std::make_unique<dealii::Vector<double>>(pre_rows * post_cols);
    // reinterpolate row by row, trans in spectral domain
    for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
      // pass a row from tmp_solution to spe_socket
      for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
        (*spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
      }
      // conduct interpolation
      spe_strans->interpolate(*spe_preslot, *spe_postslot);
      // apply hanging node constraints before
      // passing postslot into mid_solution
      simbox->spectral_frame->constraints->distribute(*spe_postslot);
      // refined Snew has not enough row lines to hold
      for (decltype(post_cols) j = 0; j < post_cols; ++j) {
        (*mid_solution)[i + j * pre_rows] = (*spe_postslot)[j];
      }
    }
    // reinterpolate col by col, trans in spatial domain
    for (decltype(post_cols) j = 0; j < post_cols; ++j) {
      // pass a col from Snew to spa_socket
      for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
        (*spa_preslot)[i] = (*mid_solution)[i + j * pre_rows];
      }
      // conduct interpolation
      spa_strans->interpolate(*spa_preslot, *spa_postslot);
      // apply hanging node constraints before
      simbox->spatial_frame->constraints->distribute(*spa_postslot);
      // passing postslot into Snew
      for (decltype(post_rows) i = 0; i < post_rows; ++i) {
        (*(this->Snew))[i + j * post_rows] = (*spa_postslot)[i];
      }
    }
  }
  // if Snew is large enough to hold intermediate info
  else {
    // reinterpolate row by row, trans in spectral domain
    for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
      // pass a row from tmp_solution to spe_socket
      for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
        (*spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
      }
      // conduct interpolation
      spe_strans->interpolate(*spe_preslot, *spe_postslot);
      // apply hanging node constraints before
      simbox->spectral_frame->constraints->distribute(*spe_postslot);
      // passing postslot into reinitiated Snew
      for (decltype(post_cols) j = 0; j < post_cols; ++j) {
        (*(this->Snew))[i + j * post_rows] = (*spe_postslot)[j];
      }
    }
    // reinterpolate col by col, trans in spatial domain
    for (decltype(post_cols) j = 0; j < post_cols; ++j) {
      // pass a col from Snew to spa_socket
      for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
        (*spa_preslot)[i] = (*(this->Snew))[i + j * post_rows];
      }
      // conduct interpolation
      spa_strans->interpolate(*spa_preslot, *spa_postslot);
      // apply hanging node constraints before
      simbox->spatial_frame->constraints->distribute(*spa_postslot);
      // pass postslot into Snew
      for (decltype(post_rows) i = 0; i < post_rows; ++i) {
        (*(this->Snew))[i + j * post_rows] = (*spa_postslot)[i];
      }
    }
  }
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::refine_spatial(
    Simbox<spa_dim, spe_dim> *simbox,
    const dealii::Vector<float> *spatial_err) {
  // cache number of rows before refinement
  auto pre_rows = this->n_rows_new;
  // cache number of cols before refinement
  auto pre_cols = this->n_cols_new;
  // cache Snew out to temporary holder
  auto tmp_solution = std::make_unique<dealii::Vector<double>>(*(this->Snew));
  // solution trans holder
  auto spa_strans = std::make_unique<dealii::SolutionTransfer<spa_dim>>(
      *(simbox->spatial_frame->dof_handler));
  // allocate spatial/spectral domain trans temporary holder
  // dof size before frame refinement
  auto spa_preslot = std::make_unique<dealii::Vector<double>>(pre_rows);
  // prepare SolutionTransfer template
  spa_strans->prepare_for_coarsening_and_refinement(*spa_preslot);
  // apply refinement to frames
  simbox->refine_spatial(spatial_err);
  // IMMEDIATELY AFTER SIMBOX REFINE
  // reinit new/old solutions
  // use dof after frame refinement
  this->new_reshape(simbox);
  this->old_reshape(simbox);
  // cache number of rows after refinement
  auto post_rows = this->n_rows_new;
  // number of cols after refinement shouldn't change
  assert(this->n_cols_new == pre_cols);
  // allocate spatial/spectral domain trans temporary holder
  // odf size after frame refinement
  auto spa_postslot = std::make_unique<dealii::Vector<double>>(post_rows);
  // reinterpolate col by col, trans in spatial domain
  for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
    // pass a col from Snew to spa_socket
    for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
      (*spa_preslot)[i] = (*tmp_solution)[i + j * pre_rows];
    }
    // conduct interpolation
    spa_strans->interpolate(*spa_preslot, *spa_postslot);
    // apply hanging node constraints before
    simbox->spatial_frame->constraints->distribute(*spa_postslot);
    // passing postslot into Snew
    for (decltype(post_rows) i = 0; i < post_rows; ++i) {
      (*(this->Snew))[i + j * post_rows] = (*spa_postslot)[i];
    }
  }
}

template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::refine_spectral(
    Simbox<spa_dim, spe_dim> *simbox,
    const dealii::Vector<float> *spectral_err) {
  // cache number of rows before refinement
  auto pre_rows = this->n_rows_new;
  // cache number of cols before refinement
  auto pre_cols = this->n_cols_new;
  // cache Snew out to temporary holder
  auto tmp_solution = std::make_unique<dealii::Vector<double>>(*(this->Snew));
  // solution trans holder
  auto spe_strans = std::make_unique<dealii::SolutionTransfer<spe_dim>>(
      *(simbox->spectral_frame->dof_handler));
  // allocate spatial/spectral domain trans temporary holder
  // dof size before frame refinement
  auto spe_preslot = std::make_unique<dealii::Vector<double>>(pre_cols);
  // prepare SolutionTransfer template
  spe_strans->prepare_for_coarsening_and_refinement(*spe_preslot);
  // apply refinement to frames
  simbox->refine_spectral(spectral_err);
  // IMMEDIATELY AFTER SIMBOX REFINE
  // reinit new/old solutions
  // use dof after frame refinement
  this->new_reshape(simbox);
  this->old_reshape(simbox);
  // cache number of cols after refinement
  auto post_cols = this->n_cols_new;
  // number of rows after refinement shouldn't change
  assert(this->n_rows_new == pre_rows);
  // allocate spatial/spectral domain trans temporary holder
  // odf size after frame refinement
  auto spe_postslot = std::make_unique<dealii::Vector<double>>(post_cols);
  // reinterpolate row by row, trans in spectral domain
  for (decltype(pre_rows) i = 0; i < pre_rows; ++i) {
    // pass a row from tmp_solution to spe_socket
    for (decltype(pre_cols) j = 0; j < pre_cols; ++j) {
      (*spe_preslot)[j] = (*tmp_solution)[i + j * pre_rows];
    }
    // conduct interpolation
    spe_strans->interpolate(*spe_preslot, *spe_postslot);
    // apply hanging node constraints before
    simbox->spectral_frame->constraints->distribute(*spe_postslot);
    // passing postslot into reinitiated Snew
    for (decltype(post_cols) j = 0; j < post_cols; ++j) {
      (*(this->Snew))[i + j * pre_rows] = (*spe_postslot)[j];
    }
  }
}

// no need for parallel
template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::pre_constraints(
    const Simbox<spa_dim, spe_dim> *simbox) {
  auto spe_slice = std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto spa_slice = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  // loop through spatial dofs
  for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i) {
    // copy from Snew
    for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j)
      (*spe_slice)[j] = this->new_el(i, j);
    simbox->spectral_frame->constraints->set_zero(*spe_slice);
    // copy back to Snew
    for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j)
      this->new_el(i, j) = (*spe_slice)[j];
  }
  // loop through spectral dofs
  for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j) {
    // copy from Snew
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i)
      (*spa_slice)[i] = this->new_el(i, j);
    simbox->spatial_frame->constraints->set_zero(*spa_slice);
    // copy back to Snew
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i)
      this->new_el(i, j) = (*spa_slice)[i];
  }
}

// no need for parallel
template <int spa_dim, int spe_dim>
void Solution<spa_dim, spe_dim>::post_constraints(
    const Simbox<spa_dim, spe_dim> *simbox) {
  auto spe_slice = std::make_unique<dealii::Vector<double>>(this->n_cols_new);
  auto spa_slice = std::make_unique<dealii::Vector<double>>(this->n_rows_new);
  // loop through spatial dofs
  for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i) {
    // copy from Snew
    for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j)
      (*spe_slice)[j] = this->new_el(i, j);
    simbox->spectral_frame->constraints->distribute(*spe_slice);
    // copy back to Snew
    for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j)
      this->new_el(i, j) = (*spe_slice)[j];
  }
  // loop through spectral dofs
  for (decltype(this->n_cols_new) j = 0; j < this->n_cols_new; ++j) {
    // copy from Snew
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i)
      (*spa_slice)[i] = this->new_el(i, j);
    simbox->spatial_frame->constraints->distribute(*spa_slice);
    // copy back to Snew
    for (decltype(this->n_rows_new) i = 0; i < this->n_rows_new; ++i)
      this->new_el(i, j) = (*spa_slice)[i];
  }
}

//------------------------------------------------------------------------------
// nested initial condition functions

template <int spa_dim, int spe_dim>
double Solution<spa_dim, spe_dim>::Spatial_initial::value(
    const dealii::Point<spa_dim, double> &, const unsigned int) const {
  return 1.;
}

template <int spa_dim, int spe_dim>
double Solution<spa_dim, spe_dim>::Spectral_initial::value(
    const dealii::Point<spe_dim, double> &, const unsigned int) const {
  return 1.;
}

// END
