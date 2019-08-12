// examples of spectral advection (energy loss) problem which is
// - defined in different dimensions
// - solved in time-dependent and time-independent approach
//
// the example problem is defined simple enough to have analytical solution
// for testing numeric precision

#include <cassert>
#include <cmath>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <advection.h>
#include <diffusion.h>
#include <frame.h>
#include <param.h>
#include <propagator.h>
#include <simbox.h>
#include <solution.h>
#include <source.h>
#include <system.h>

//------------------------------------------------------------------------------
// customized advection tensor

template <int spa_dim, int spe_dim>
class Advection_tmp final : public Advection<spa_dim, spe_dim> {
public:
  Advection_tmp(const Param *);
  virtual ~Advection_tmp() = default;
  Advection_tmp(const Advection_tmp &a) : Advection<spa_dim, spe_dim>(a) {
    this->alpha = a.alpha;
    this->beta = a.beta;
    this->eta = a.eta;
    this->nz = a.nz;
    this->nx = a.nx;
    this->ny = a.ny;
    this->sz = a.sz;
    this->sx = a.sx;
    this->sy = a.sy;
    this->L0 = a.L0;
    this->L1 = a.L1;
    this->L2 = a.L2;
    this->q0_min = a.q0_min;
    this->q1_min = a.q1_min;
    this->q2_min = a.q2_min;
  }
  virtual Advection_tmp *clone() const { return new Advection_tmp(*this); }
  dealii::Tensor<1, spe_dim, double>
  Aqq(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &,
      const double &step_time = 0) const override;

  double alpha, beta, eta, nz, nx, ny, sz, sx, sy;
  double L0, L1, L2, q0_min, q1_min, q2_min;
};

template <int spa_dim, int spe_dim>
Advection_tmp<spa_dim, spe_dim>::Advection_tmp(const Param *par) {
  this->alpha = -1.;
  this->beta = -1.;
  this->eta = -1.;
  this->nz = -1.5;
  this->nx = -1.5;
  this->ny = -1.5;
  this->sz = -2.2;
  this->sx = -2.2;
  this->sy = -2.2;
  this->q0_min = par->grid_set.q1_min;
  this->q1_min = par->grid_set.q2_min;
  this->q2_min = par->grid_set.q3_min;
}

// customized advection modelling
template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double>
Advection_tmp<spa_dim, spe_dim>::Aqq(const dealii::Point<spa_dim, double> &,
                                     const dealii::Point<spe_dim, double> &q,
                                     const double &) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  // E[0]
  tmp[dealii::TableIndices<1>(0)] =
      this->alpha * std::exp(this->nz * (q[0] - this->q0_min));
  if (spe_dim > 1) {
    // E[1]
    tmp[dealii::TableIndices<1>(1)] =
        this->beta * std::exp(this->nx * (q[1] - this->q1_min));
    if (spe_dim > 2) {
      // E[2]
      tmp[dealii::TableIndices<1>(2)] =
          this->eta * std::exp(this->ny * (q[2] - this->q2_min));
    }
  }
  return tmp;
}

//------------------------------------------------------------------------------
// customized source

template <int spa_dim, int spe_dim>
class Source_tmp final : public Source<spa_dim, spe_dim> {
public:
  Source_tmp(const Param *);
  virtual ~Source_tmp() = default;
  Source_tmp(const Source_tmp &s) : Source<spa_dim, spe_dim>(s) {
    this->alpha = s.alpha;
    this->beta = s.beta;
    this->eta = s.eta;
    this->nz = s.nz;
    this->nx = s.nx;
    this->ny = s.ny;
    this->sz = s.sz;
    this->sx = s.sx;
    this->sy = s.sy;
    this->L0 = s.L0;
    this->L1 = s.L1;
    this->L2 = s.L2;
    this->q0_min = s.q0_min;
    this->q1_min = s.q1_min;
    this->q2_min = s.q2_min;
  }
  virtual Source_tmp *clone() const { return new Source_tmp(*this); }
  double value(const dealii::Point<spa_dim, double> &,
               const dealii::Point<spe_dim, double> &,
               const double &step_time = 0) const override;

  double alpha, beta, eta, nz, nx, ny, sz, sx, sy;
  double L0, L1, L2, q0_min, q1_min, q2_min;
};

template <int spa_dim, int spe_dim>
Source_tmp<spa_dim, spe_dim>::Source_tmp(const Param *par) {
  this->alpha = -1.;
  this->beta = -1.;
  this->eta = -1.;
  this->nz = -1.5;
  this->nx = -1.5;
  this->ny = -1.5;
  this->sz = -2.2;
  this->sx = -2.2;
  this->sy = -2.2;
  this->q0_min = par->grid_set.q1_min;
  this->q1_min = par->grid_set.q2_min;
  this->q2_min = par->grid_set.q3_min;
  this->L0 = par->grid_set.q1_max - this->q0_min;
  this->L1 = par->grid_set.q2_max - this->q1_min;
  this->L2 = par->grid_set.q3_max - this->q2_min;
}

// customized source modelling
template <int spa_dim, int spe_dim>
double
Source_tmp<spa_dim, spe_dim>::value(const dealii::Point<spa_dim, double> &,
                                    const dealii::Point<spe_dim, double> &q,
                                    const double &) const {
  const double prefact0{std::exp(this->q0_min) /
                        (this->alpha * (1 + this->sz))};
  const double dz{q[0] - this->q0_min};
  const double u0{prefact0 *
                  (std::exp((1 + this->sz - this->nz) * dz) -
                   std::exp((1 + this->sz) * this->L0 - this->nz * dz))};
  const double f0{std::exp(this->sz * dz)};
  double tmp = f0;
  if (spe_dim > 1) {
    const double prefact1{std::exp(this->q1_min) /
                          (this->beta * (1 + this->sx))};
    const double dx{q[1] - this->q1_min};
    const double f1{std::exp(this->sx * dx)};
    const double u1{prefact1 *
                    (std::exp((1 + this->sx - this->nx) * dx) -
                     std::exp((1 + this->sx) * this->L1 - this->nx * dx))};
    tmp = tmp * u1 + u0 * f1;
    if (spe_dim > 2) {
      const double prefact2{std::exp(this->q2_min) /
                            (this->eta * (1 + this->sy))};
      const double dy{q[2] - this->q2_min};
      const double f2{std::exp(this->sy * dy)};
      const double u2{prefact2 *
                      (std::exp((1 + this->sy - this->ny) * dy) -
                       std::exp((1 + this->sy) * this->L2 - this->ny * dy))};
      tmp = tmp * u2 + u0 * u1 * f2;
    }
  }
  return tmp;
}

//------------------------------------------------------------------------------
// customized Simbox

template <int spa_dim, int spe_dim>
class Simbox_tmp final : public Simbox<spa_dim, spe_dim> {
public:
  Simbox_tmp() = default;
  Simbox_tmp(const Param *);
  virtual ~Simbox_tmp() = default;
};

template <int spa_dim, int spe_dim>
Simbox_tmp<spa_dim, spe_dim>::Simbox_tmp(const Param *par) {
  this->spatial_frame = std::make_unique<Frame_freespatial<spa_dim>>(par);
  this->spectral_frame = std::make_unique<dGFrame_spectral<spe_dim>>(par);
  this->sparsity = std::make_unique<dealii::SparsityPattern>();
  this->dsp = std::make_unique<dealii::DynamicSparsityPattern>();
}

//------------------------------------------------------------------------------
// customized System

// spectral advection (energy loss) System uses discontinuous Dalerkin method
// for spectral frame which involves surface integration on both internal and
// external surface elements fitine-element discontinuous Galerkin can be
// considered as a hybrid of fintie-element and finite-volume
template <int spa_dim, int spe_dim>
class System_tmp final : public System<spa_dim, spe_dim> {
public:
  System_tmp() = default;
  System_tmp(const Param *);
  virtual ~System_tmp() = default;
  // copy ctor
  System_tmp(const System_tmp<spa_dim, spe_dim> &s)
      : System<spa_dim, spe_dim>(s) {}
  // virtual copy
  virtual System_tmp *clone() const { return new System_tmp(*this); }
  // nested operator class
  class Operator : public System<spa_dim, spe_dim>::Operator {
  public:
    Operator() = default;
    virtual ~Operator() = default;
    virtual Operator *clone() const { return new Operator(*this); }
    // configure Mxq
    void init(System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
              const double &step_time = 0) override;

  protected:
    dealii::Tensor<2, spe_dim, double>
    T(const dealii::Point<spe_dim, double> &) const;
    dealii::Tensor<1, spe_dim, double>
    dT(const dealii::Point<spe_dim, double> &) const;
    // assemble face terms for init function
    void assemble_spectral_face(
        System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
        const dealii::FEValues<spa_dim> *, const unsigned int &,
        const dealii::FEFaceValuesBase<spe_dim> *,
        const dealii::FEFaceValuesBase<spe_dim> *,
        std::vector<dealii::types::global_dof_index> *,
        std::vector<dealii::types::global_dof_index> *,
        dealii::FullMatrix<double> *, dealii::FullMatrix<double> *,
        dealii::FullMatrix<double> *, dealii::FullMatrix<double> *,
        const double &step_time = 0);
  };
  // nested rhs class
  class RHS : public System<spa_dim, spe_dim>::RHS {
  public:
    RHS() = default;
    virtual ~RHS() = default;
    virtual RHS *clone() const { return new RHS(*this); }
    // initialize/fill system_rhs with value info
    void init(System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
              const double &step_time = 0) override;
  };
};

template <int spa_dim, int spe_dim>
System_tmp<spa_dim, spe_dim>::System_tmp(const Param *par) {
  // allocate sparse matrix
  this->Mx = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  // allocate vector
  this->Rxq = std::make_unique<dealii::Vector<double>>();
  this->Rx = std::make_unique<dealii::Vector<double>>();
  this->Rq = std::make_unique<dealii::Vector<double>>();
  // rebind Operator and RHS holder
  this->op =
      std::make_unique<typename System_tmp<spa_dim, spe_dim>::Operator>();
  this->rhs = std::make_unique<typename System_tmp<spa_dim, spe_dim>::RHS>();
  // physics
  this->diffusion = std::make_unique<Diffusion<spa_dim, spe_dim>>();
  this->advection = std::make_unique<Advection_tmp<spa_dim, spe_dim>>(par);
  this->source = std::make_unique<Source_tmp<spa_dim, spe_dim>>(par);
}

// geometric tensor
template <int spa_dim, int spe_dim>
dealii::Tensor<2, spe_dim, double> System_tmp<spa_dim, spe_dim>::Operator::T(
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<2, spe_dim, double> tmp;
  tmp = 0;
  tmp[dealii::TableIndices<2>(0, 0)] = std::exp(-q[0]);
  if (spe_dim > 1) {
    tmp[dealii::TableIndices<2>(1, 1)] = std::exp(-q[1]);
    if (spe_dim > 2) {
      tmp[dealii::TableIndices<2>(2, 2)] = std::exp(-q[2]);
    }
  }
  return tmp;
}

// differentiated geometric tensor
template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> System_tmp<spa_dim, spe_dim>::Operator::dT(
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  tmp[dealii::TableIndices<1>(0)] = -std::exp(-q[0]);
  if (spe_dim > 1) {
    tmp[dealii::TableIndices<1>(1)] = -std::exp(-q[1]);
    if (spe_dim > 2) {
      tmp[dealii::TableIndices<1>(2)] = -std::exp(-q[2]);
    }
  }
  return tmp;
}

template <int spa_dim, int spe_dim>
void System_tmp<spa_dim, spe_dim>::Operator::init(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const double &step_time) {
  // auxiliary objects needed for conduct discretized integration
  auto spatial_quadrature_formula = std::make_unique<dealii::QGauss<spa_dim>>(
      simbox->spatial_frame->fe->degree + 1);
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      simbox->spectral_frame->fe->degree + 1);
  auto spectral_face_quadrature_formula =
      std::make_unique<dealii::QGauss<spe_dim - 1>>(
          simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  // update_gradients is critical
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // fe_values in spectral domain
  // update_values is critical
  auto spectral_fev = std::make_unique<dealii::FEValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_quadrature_formula,
      dealii::update_values | dealii::update_gradients |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  // fe face values
  auto spectral_fefv = std::make_unique<dealii::FEFaceValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_face_quadrature_formula,
      dealii::update_values | dealii::update_normal_vectors |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  // fe subface values
  auto spectral_fesfv = std::make_unique<dealii::FESubfaceValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_face_quadrature_formula,
      dealii::update_values | dealii::update_normal_vectors |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  // fe neighbor face values
  auto spectral_fenfv = std::make_unique<dealii::FEFaceValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_face_quadrature_formula,
      dealii::update_values);
  // spatial/spectral domain dofs per cell
  const unsigned int spatial_dpc = spatial_fev->dofs_per_cell;
  const unsigned int spectral_dpc = spectral_fev->dofs_per_cell;
  // spatial/spectral domain quadrature points per cell
  const unsigned int spatial_q_points = spatial_quadrature_formula->size();
  const unsigned int spectral_q_points = spectral_quadrature_formula->size();
  const unsigned int spectral_face_q_points =
      spectral_face_quadrature_formula->size();
  // local 2 global indeces holder
  auto spatial_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spatial_dpc);
  auto spectral_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);
  auto spectral_l2g_neighbor =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);
  // allocate temporary local (per-cell) matrix holder
  // allocate fullmatrices
  auto cell_Mx =
      std::make_unique<dealii::FullMatrix<double>>(spatial_dpc, spatial_dpc);
  auto cell_Mq_uivi =
      std::make_unique<dealii::FullMatrix<double>>(spectral_dpc, spectral_dpc);
  auto cell_Mq_uive =
      std::make_unique<dealii::FullMatrix<double>>(spectral_dpc, spectral_dpc);
  auto cell_Mq_uevi =
      std::make_unique<dealii::FullMatrix<double>>(spectral_dpc, spectral_dpc);
  auto cell_Mq_ueve =
      std::make_unique<dealii::FullMatrix<double>>(spectral_dpc, spectral_dpc);
  // allocate memory for Mxq
  system->Mxq->reinit(*(simbox->sparsity));
  // fill Mxq with values
  // (integration with base functions over two domains)
  // loop over spatial/spectral domain cells
#ifdef _OPENMP
  system->omp_cell_distribute(simbox);
  for (auto spatial_cell = system->it_start; spatial_cell != system->it_end;
       ++spatial_cell)
#else
  for (const auto &spatial_cell :
       simbox->spatial_frame->dof_handler->active_cell_iterators())
#endif
  {
    spatial_fev->reinit(spatial_cell);
    // (from per-cell indeces to global indeces)
#ifndef NDEBUG
    assert(spatial_cell->active());
#endif
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (const auto &spectral_cell :
         simbox->spectral_frame->dof_handler->active_cell_iterators()) {
#ifdef VERBOSE
      if (spatial_cell->index() == 0) {
        std::cout << std::endl
                  << "-------------------------------------------" << std::endl
                  << "in Operator assembling: " << std::endl
                  << "target cell: " << spectral_cell->index()
                  << "\t lv: " << spectral_cell->level() << std::endl;
      }
#endif
      spectral_fev->reinit(spectral_cell);
      // (from per-cell indeces to global indeces)
#ifndef NDEBUG
      assert(spectral_cell->active());
#endif
      spectral_cell->get_dof_indices(*spectral_l2g);
      // discrete integration over spectral domain per cell
      for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
           ++spatial_qid) {
        // prepare Mx
        for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
          for (dealii::types::global_dof_index j = 0; j < spatial_dpc; ++j) {
            cell_Mx->set(i, j,
                         spatial_fev->shape_value(i, spatial_qid) *
                             spatial_fev->shape_value(j, spatial_qid) *
                             spatial_fev->JxW(spatial_qid));
          } // j
        }   // i
        // (convert local full matrix to global sparse matrix cacher)
        system->Mx->reinit(*(simbox->spatial_frame->sparsity));
        simbox->spatial_frame->constraints->distribute_local_to_global(
            *cell_Mx, *spatial_l2g, *(system->Mx));
        // accumulate volume terms
        for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
             ++spectral_qid) {
          const dealii::Tensor<1, spe_dim, double> advection_coefficient{
              system->advection->Aqq(
                  spatial_fev->quadrature_point(spatial_qid),
                  spectral_fev->quadrature_point(spectral_qid), step_time)};
          const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
              this->T(spectral_fev->quadrature_point(spectral_qid))};
          const dealii::Tensor<1, spe_dim, double> geometric_div_coefficient{
              this->dT(spectral_fev->quadrature_point(spectral_qid))};
          for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
               ++alpha) {
            for (dealii::types::global_dof_index beta = 0; beta < spectral_dpc;
                 ++beta) {
              cell_Mq_uivi->set(
                  alpha, beta,
                  -1. *
                      (spectral_fev->shape_value(alpha, spectral_qid) *
                           dealii::scalar_product(geometric_div_coefficient,
                                                  advection_coefficient) +
                       dealii::scalar_product(
                           spectral_fev->shape_grad(alpha, spectral_qid),
                           geometric_coefficient * advection_coefficient)) *
                      spectral_fev->shape_value(beta, spectral_qid) *
                      spectral_fev->JxW(spectral_qid));
            } // beta
          }   // alpha
          // (convert local full matrix to global sparse matrix cacher)
          system->Mq->reinit(*(simbox->spectral_frame->sparsity));
          simbox->spectral_frame->constraints->distribute_local_to_global(
              *cell_Mq_uivi, *spectral_l2g, *(system->Mq));
          // accumulate to Mxq
          system->Operator_Kronecker_accumulate(simbox);
        } // spectral_qid
        // accumulate face terms
        for (unsigned int face_id = 0;
             face_id < dealii::GeometryInfo<spe_dim>::faces_per_cell;
             ++face_id) {
#ifdef VERBOSE
          if (spatial_cell->index() == 0 and spatial_qid == 0) {
            std::cout << std::endl << "at face: " << face_id << std::endl;
          }
#endif
          typename dealii::DoFHandler<spe_dim>::face_iterator face =
              spectral_cell->face(face_id);
          // at lower boundary faces
          if (face->at_boundary()) {
#ifdef VERBOSE
            if (spatial_cell->index() == 0 and spatial_qid == 0) {
              std::cout << std::endl
                        << "on boundary" << std::endl
                        << "---------------------" << std::endl;
            }
#endif
            if (face->boundary_id() % 2 == 0) {
              spectral_fefv->reinit(spectral_cell, face_id);
              // (accumulate boundary terms)
              for (unsigned int spectral_qid = 0;
                   spectral_qid < spectral_face_q_points; ++spectral_qid) {
                const dealii::Tensor<1, spe_dim, double> advection_coefficient{
                    system->advection->Aqq(
                        spatial_fev->quadrature_point(spatial_qid),
                        spectral_fefv->quadrature_point(spectral_qid),
                        step_time)};
                const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
                    this->T(spectral_fefv->quadrature_point(spectral_qid))};
                for (dealii::types::global_dof_index alpha = 0;
                     alpha < spectral_dpc; ++alpha) {
                  for (dealii::types::global_dof_index beta = 0;
                       beta < spectral_dpc; ++beta) {
                    cell_Mq_uivi->set(
                        alpha, beta,
                        spectral_fefv->shape_value(alpha, spectral_qid) *
                            dealii::scalar_product(
                                geometric_coefficient * advection_coefficient,
                                spectral_fefv->normal_vector(spectral_qid)) *
                            spectral_fefv->shape_value(beta, spectral_qid) *
                            spectral_fefv->JxW(spectral_qid));
                  } // beta
                }   // alpha
                // (convert local full matrix to global sparse matrix cacher)
                system->Mq->reinit(*(simbox->spectral_frame->sparsity));
                simbox->spectral_frame->constraints->distribute_local_to_global(
                    *cell_Mq_uivi, *spectral_l2g, *(system->Mq));
                // accumulate to Mxq
                system->Operator_Kronecker_accumulate(simbox);
              } // spectral_qid
            }   // pick lower boundary
          }     // at boundary
          // internal faces
          else {
            typename dealii::DoFHandler<spe_dim>::cell_iterator neighbor =
                spectral_cell->neighbor(face_id);
#ifdef VERBOSE
            if (spatial_cell->index() == 0 and spatial_qid == 0) {
              std::cout << std::endl
                        << "neighbor cell: " << neighbor->index()
                        << "\t lv: " << neighbor->level() << std::endl;
            }
#endif
            if (face->has_children()) {
#ifdef VERBOSE
              if (spatial_cell->index() == 0 and spatial_qid == 0) {
                std::cout << std::endl << "face has children" << std::endl;
              }
#endif
              const unsigned int neighbor_face_id =
                  spectral_cell->neighbor_face_no(face_id);
              for (unsigned int subface_id = 0;
                   subface_id < face->number_of_children(); ++subface_id) {
                typename dealii::DoFHandler<spe_dim>::cell_iterator
                    neighbor_child = spectral_cell->neighbor_child_on_subface(
                        face_id, subface_id);
                spectral_fesfv->reinit(spectral_cell, face_id, subface_id);
                spectral_fenfv->reinit(neighbor_child, neighbor_face_id);
#ifndef NDEBUG
                assert(!neighbor_child->has_children());
#endif
                neighbor_child->get_dof_indices(*spectral_l2g_neighbor);
                this->assemble_spectral_face(
                    system, simbox, spatial_fev.get(), spatial_qid,
                    spectral_fesfv.get(), spectral_fenfv.get(),
                    spectral_l2g.get(), spectral_l2g_neighbor.get(),
                    cell_Mq_uivi.get(), cell_Mq_uevi.get(), cell_Mq_uive.get(),
                    cell_Mq_ueve.get(), step_time);
              }                            // subface_id
            }                              // face has children
            else if (neighbor->active()) { // DISCARD NON-ACTIVE NEIGHBOR
#ifdef VERBOSE
              if (spatial_cell->index() == 0 and spatial_qid == 0) {
                std::cout << std::endl << "neighbor is active" << std::endl;
              }
#endif
              if (spe_dim > 1 and spectral_cell->neighbor_is_coarser(face_id))
                continue; // SPECIAL FOR DIM!=1
              // cells at higher level take the task
              // or at same level but lower index
              // in 1 dimension a cell cannot "see" neighbor at higher
              // refinement level and no face has children anymore
              if (neighbor->level() < spectral_cell->level() or
                  (neighbor->level() == spectral_cell->level() and
                   neighbor->index() > spectral_cell->index())) {
                const unsigned int neighbor_face_id =
                    spectral_cell->neighbor_face_no(face_id);
                spectral_fefv->reinit(spectral_cell, face_id);
                spectral_fenfv->reinit(neighbor, neighbor_face_id);
#ifdef VERBOSE
                if (spatial_cell->index() == 0 and spatial_qid == 0) {
                  std::cout << std::endl
                            << "neighbor " << neighbor->index()
                            << " paired on face: " << neighbor_face_id
                            << std::endl
                            << "-------------------------------------------"
                            << std::endl;
                }
#endif
#ifndef NDEBUG
                assert(neighbor->active());
#endif
                neighbor->get_dof_indices(*spectral_l2g_neighbor);
                this->assemble_spectral_face(
                    system, simbox, spatial_fev.get(), spatial_qid,
                    spectral_fefv.get(), spectral_fenfv.get(),
                    spectral_l2g.get(), spectral_l2g_neighbor.get(),
                    cell_Mq_uivi.get(), cell_Mq_uevi.get(), cell_Mq_uive.get(),
                    cell_Mq_ueve.get(), step_time);
              } // selected order
            }   // face has no children
          }     // internal faces
        }       // face_id
      }         // spatial_qid
    }           // spectral_cell loop
  }             // spatial_cell loop
}

template <int spa_dim, int spe_dim>
void System_tmp<spa_dim, spe_dim>::Operator::assemble_spectral_face(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const dealii::FEValues<spa_dim> *fev, const unsigned int &spatial_qid,
    const dealii::FEFaceValuesBase<spe_dim> *fefv,
    const dealii::FEFaceValuesBase<spe_dim> *fenfv,
    std::vector<dealii::types::global_dof_index> *l2g,
    std::vector<dealii::types::global_dof_index> *l2gn,
    dealii::FullMatrix<double> *uivi, dealii::FullMatrix<double> *uevi,
    dealii::FullMatrix<double> *uive, dealii::FullMatrix<double> *ueve,
    const double &step_time) {
  const unsigned int q_points = fefv->n_quadrature_points;
  const unsigned int master_dpc = fefv->dofs_per_cell;
  const unsigned int neighbor_dpc = fenfv->dofs_per_cell;
  for (unsigned int spectral_qid = 0; spectral_qid < q_points; ++spectral_qid) {
    *uivi = 0;
    *uevi = 0;
    *uive = 0;
    *ueve = 0;
    const dealii::Tensor<1, spe_dim, double> advection_coefficient{
        system->advection->Aqq(fev->quadrature_point(spatial_qid),
                               fefv->quadrature_point(spectral_qid),
                               step_time)};
    const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
        this->T(fefv->quadrature_point(spectral_qid))};
    const double flux =
        dealii::scalar_product(geometric_coefficient * advection_coefficient,
                               fefv->normal_vector(spectral_qid));
    if (flux > 0) {
      for (unsigned int i = 0; i < master_dpc; ++i) {
        for (unsigned int j = 0; j < master_dpc; ++j) {
          uivi->set(i, j,
                    flux * fefv->shape_value(j, spectral_qid) *
                        fefv->shape_value(i, spectral_qid) *
                        fefv->JxW(spectral_qid));
        } // j
      }   // i
      for (unsigned int k = 0; k < neighbor_dpc; ++k) {
        for (unsigned int j = 0; j < master_dpc; ++j) {
          uive->set(k, j,
                    -flux * fefv->shape_value(j, spectral_qid) *
                        fenfv->shape_value(k, spectral_qid) *
                        fefv->JxW(spectral_qid));
        } // j
      }   // k
    } else {
      for (unsigned int i = 0; i < master_dpc; ++i) {
        for (unsigned int l = 0; l < neighbor_dpc; ++l) {
          uevi->set(i, l,
                    flux * fenfv->shape_value(l, spectral_qid) *
                        fefv->shape_value(i, spectral_qid) *
                        fefv->JxW(spectral_qid));
        } // l
      }   // i
      for (unsigned int k = 0; k < neighbor_dpc; ++k) {
        for (unsigned int l = 0; l < neighbor_dpc; ++l) {
          ueve->set(k, l,
                    -flux * fenfv->shape_value(l, spectral_qid) *
                        fenfv->shape_value(k, spectral_qid) *
                        fefv->JxW(spectral_qid));
        } // l
      }   // k
    }     // flux in
    // (copy face terms to system Mq)
    system->Mq->reinit(*(simbox->spectral_frame->sparsity));
    for (dealii::types::global_dof_index alpha = 0; alpha < master_dpc;
         ++alpha) {
      for (dealii::types::global_dof_index beta = 0; beta < master_dpc;
           ++beta) {
        system->Mq->set(l2g->at(alpha), l2g->at(beta), (*uivi)(alpha, beta));
        system->Mq->add(l2g->at(alpha), l2gn->at(beta), (*uevi)(alpha, beta));
        system->Mq->add(l2gn->at(alpha), l2g->at(beta), (*uive)(alpha, beta));
        system->Mq->add(l2gn->at(alpha), l2gn->at(beta), (*ueve)(alpha, beta));
      } // beta
    }   // alpha
    // accumulate to Mxq
    system->Operator_Kronecker_accumulate(simbox);
  } // spectral_qid
}

template <int spa_dim, int spe_dim>
void System_tmp<spa_dim, spe_dim>::RHS::init(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const double &step_time) {
  // auxiliary objects needed for conduct discretized integration
  auto spatial_quadrature_formula = std::make_unique<dealii::QGauss<spa_dim>>(
      simbox->spatial_frame->fe->degree + 1);
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // fe_values in spectral domain
  auto spectral_fev = std::make_unique<dealii::FEValues<spe_dim>>(
      *(simbox->spectral_frame->fe), *spectral_quadrature_formula,
      dealii::update_values | dealii::update_quadrature_points |
          dealii::update_JxW_values);
  // spatial/spectral domain dofs per cell
  const unsigned int spatial_dpc = spatial_fev->dofs_per_cell;
  const unsigned int spectral_dpc = spectral_fev->dofs_per_cell;
  // spatial/spectral domain quadrature points per cell
  const unsigned int spatial_q_points = spatial_quadrature_formula->size();
  const unsigned int spectral_q_points = spectral_quadrature_formula->size();
  // local 2 global indeces holder
  auto spatial_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spatial_dpc);
  auto spectral_l2g =
      std::make_unique<std::vector<dealii::types::global_dof_index>>(
          spectral_dpc);
  // allocate temporary local (per-cell) spatial RHS and spectral RHS vector
  // holder
  auto spatial_cell_rhs = std::make_unique<dealii::Vector<double>>(spatial_dpc);
  auto spectral_cell_rhs =
      std::make_unique<dealii::Vector<double>>(spectral_dpc);
  // allocate global system RHS
  system->Rxq->reinit(simbox->spatial_frame->dof_handler->n_dofs() *
                      simbox->spectral_frame->dof_handler->n_dofs());

  // fill RHS "matrix" with values
  // (integration with base functions over two domains)
  // loop over spatial/spectral domain cells
#ifdef _OPENMP
  system->omp_cell_distribute(simbox);
  for (auto spatial_cell = system->it_start; spatial_cell != system->it_end;
       ++spatial_cell)
#else
  for (const auto &spatial_cell :
       simbox->spatial_frame->dof_handler->active_cell_iterators())
#endif
  {
    spatial_fev->reinit(spatial_cell);
    // from per-cell indeces to global indeces
#ifndef NDEBUG
    assert(spatial_cell->active());
#endif
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (const auto &spectral_cell :
         simbox->spectral_frame->dof_handler->active_cell_iterators()) {
      spectral_fev->reinit(spectral_cell);
      // from per-cell indeces to global indeces
#ifndef NDEBUG
      assert(spectral_cell->active());
#endif
      spectral_cell->get_dof_indices(*spectral_l2g);
      // discrete integration over spatial domain per cell
      for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
           ++spatial_qid) {
        // prepare spatial cell rhs
        for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
          (*spatial_cell_rhs)[i] = spatial_fev->shape_value(i, spatial_qid) *
                                   spatial_fev->JxW(spatial_qid);
        } // i
        // (clean cacher)
        system->Rx->reinit(simbox->spatial_frame->dof_handler->n_dofs());
        // (convert local vector to global vector, with per-domain constraints
        // applied)
        simbox->spatial_frame->constraints->distribute_local_to_global(
            *spatial_cell_rhs, *spatial_l2g, *(system->Rx));
        // discrete integration over spectral domain per cell
        for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
             ++spectral_qid) {
          const double coefficient{system->source->value(
              spatial_fev->quadrature_point(spatial_qid),
              spectral_fev->quadrature_point(spectral_qid), step_time)};
          // prepare spectral cell rhs
          for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
               ++alpha) {
            (*spectral_cell_rhs)[alpha] =
                coefficient * spectral_fev->shape_value(alpha, spectral_qid) *
                spectral_fev->JxW(spectral_qid);
          } // alpha
          // (clean cacher)
          system->Rq->reinit(simbox->spectral_frame->dof_handler->n_dofs());
          // (convert local vector to global vector, with per-domain constraints
          // applied)
          simbox->spectral_frame->constraints->distribute_local_to_global(
              *spectral_cell_rhs, *spectral_l2g, *(system->Rq));
          // accumulate system rhs with Kronecker product
          system->RHS_Kronecker_accumulate();
        } // spectral_qid
      }   // spatial_qid
    }     // spectral_cell loop
  }       // spatial_cell loop
}

//------------------------------------------------------------------------------
// customized Propagator

template <int spa_dim, int spe_dim>
class Propagator_tmp final : public Propagator<spa_dim, spe_dim> {
public:
  Propagator_tmp() = default;
  Propagator_tmp(const Param *);
  virtual ~Propagator_tmp() = default;
  void solve_single_step() override;
  // spatial interpolation position
  dealii::Point<spa_dim, double> spatial_ref;
  // calculating L2 error w.r.t. analytical result
  double spectral_L2err() const;
  // analytical solution
  class analytical_solution : public dealii::Function<spe_dim, double> {
  public:
    analytical_solution() = default;
    analytical_solution(const Param *);
    virtual ~analytical_solution() = default;
    double value(const dealii::Point<spe_dim, double> &,
                 const unsigned int component = 0) const override;

  protected:
    double alpha, beta, eta, nz, nx, ny, sz, sx, sy;
    double L0, L1, L2, q0_min, q1_min, q2_min;
  };
  std::unique_ptr<
      typename Propagator_tmp<spa_dim, spe_dim>::analytical_solution>
      baseline;
};

template <int spa_dim, int spe_dim>
Propagator_tmp<spa_dim, spe_dim>::Propagator_tmp(const Param *par) {
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
  this->simbox = std::make_unique<Simbox_tmp<spa_dim, spe_dim>>(par);
  this->solution = std::make_unique<Solution<spa_dim, spe_dim>>(par);
  this->system = std::make_unique<System_tmp<spa_dim, spe_dim>>(par);
  this->baseline = std::make_unique<
      typename Propagator_tmp<spa_dim, spe_dim>::analytical_solution>(par);
  // setup spectral interpolation position
  this->spatial_ref[0] = 0.5 * (par->grid_set.x1_max + par->grid_set.x1_min);
  if (spa_dim > 1) {
    this->spatial_ref[1] = 0.5 * (par->grid_set.x2_max + par->grid_set.x2_min);
    if (spa_dim > 2) {
      this->spatial_ref[2] =
          0.5 * (par->grid_set.x3_max + par->grid_set.x3_min);
    }
  }
}

template <int spa_dim, int spe_dim>
void Propagator_tmp<spa_dim, spe_dim>::solve_single_step() {
  // direct solver
  auto solver_control = std::make_unique<dealii::SparseDirectUMFPACK>();
  solver_control->initialize(*(this->system->Mxq));
  solver_control->vmult(*(this->solution->Snew), *(this->system->Rxq));
  // only need to redistribute constraints to solution
  this->solution->post_constraints(this->simbox.get());
  /*
   // BiCGStab solver
   auto solver_control = std::make_unique<dealii::SolverControl>
   (this->iteration, this->tolerance * this->system->Rxq->l2_norm(), false,
   false);
   auto bicg = std::make_unique<dealii::SolverBicgstab<>> (*solver_control);
   bicg->solve (*(this->system->Mxq),
   *(this->solution->Snew),
   *(this->system->Rxq),
   dealii::PreconditionIdentity());
   this->solution->post_constraints (this->simbox.get());
   #if !defined(NDEBUG) || defined(VERBOSE)
   std::cout<< "   " << solver_control->last_step()
   << " BiCGStab iterations." << std::endl;
   #endif
   */
}

template <int spa_dim, int spe_dim>
Propagator_tmp<spa_dim, spe_dim>::analytical_solution::analytical_solution(
    const Param *par) {
  this->alpha = -1.;
  this->beta = -1.;
  this->eta = -1.;
  this->nz = -1.5;
  this->nx = -1.5;
  this->ny = -1.5;
  this->sz = -2.2;
  this->sx = -2.2;
  this->sy = -2.2;
  this->q0_min = par->grid_set.q1_min;
  this->q1_min = par->grid_set.q2_min;
  this->q2_min = par->grid_set.q3_min;
  this->L0 = par->grid_set.q1_max - this->q0_min;
  this->L1 = par->grid_set.q2_max - this->q1_min;
  this->L2 = par->grid_set.q3_max - this->q2_min;
}

template <int spa_dim, int spe_dim>
double Propagator_tmp<spa_dim, spe_dim>::analytical_solution::value(
    const dealii::Point<spe_dim, double> &q, const unsigned int) const {
  const double prefact0{std::exp(this->q0_min) /
                        (this->alpha * (1 + this->sz))};
  const double dz{q[0] - this->q0_min};
  double tmp{prefact0 * (std::exp((1 + this->sz - this->nz) * dz) -
                         std::exp((1 + this->sz) * this->L0 - this->nz * dz))};
  if (spe_dim > 1) {
    const double prefact1{std::exp(this->q1_min) /
                          (this->beta * (1 + this->sx))};
    const double dx{q[1] - this->q1_min};
    tmp *= prefact1 * (std::exp((1 + this->sx - this->nx) * dx) -
                       std::exp((1 + this->sx) * this->L1 - this->nx * dx));
    if (spe_dim > 2) {
      const double prefact2{std::exp(this->q2_min) /
                            (this->eta * (1 + this->sy))};
      const double dy{q[2] - this->q2_min};
      tmp *= prefact2 * (std::exp((1 + this->sy - this->ny) * dy) -
                         std::exp((1 + this->sy) * this->L2 - this->ny * dy));
    }
  }
  return tmp;
}

template <int spa_dim, int spe_dim>
double Propagator_tmp<spa_dim, spe_dim>::spectral_L2err() const {
  // interpolate at this->spectral_ref
  const unsigned int spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const unsigned int spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spe_slice = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto tmp_spatial = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto field_spatial =
      std::make_unique<dealii::Functions::FEFieldFunction<spa_dim>>(
          *(this->simbox->spatial_frame->dof_handler), *tmp_spatial);
  auto c_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spatial_frame->dof_handler), this->spatial_ref);
  field_spatial->set_active_cell(c_id);
  for (unsigned int j = 0; j < spectral_dofs; ++j) {
    for (unsigned int i = 0; i < spatial_dofs; ++i)
      (*tmp_spatial)[i] = this->solution->new_el(i, j);
    (*spe_slice)[j] = field_spatial->value(this->spatial_ref);
  }
  // evaluate L2 err with spe_slice
  auto cellwise_err = std::make_unique<dealii::Vector<double>>(
      this->simbox->spectral_frame->triangulation->n_active_cells());
  dealii::VectorTools::integrate_difference(
      *(this->simbox->spectral_frame->dof_handler), *spe_slice,
      *(this->baseline), *cellwise_err,
      dealii::QIterated<spe_dim>(dealii::QTrapez<1>(),
                                 this->simbox->spectral_frame->fe->degree + 2),
      dealii::VectorTools::L2_norm);
  return dealii::VectorTools::compute_global_error(
      *(this->simbox->spectral_frame->triangulation), *cellwise_err,
      dealii::VectorTools::L2_norm);
}

//------------------------------------------------------------------------------
// routines

// time-independent approach
// spectral domain is set as 1D
void static_11routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.q2_max = 1;
  test_par->grid_set.q2_min = -1;
  test_par->grid_set.q3_max = 1;
  test_par->grid_set.q3_min = -1;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = true;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spectral_L2err() << std::endl;
    // test_prop_rg->spectral_snapshot (dealii::Point<1,double>(0),
    //"1d_tmp_rg_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // adaptive refine
  std::cout << "1D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "adaptive_gradient";
  auto test_prop_ra = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spectral_L2err() << std::endl;
    // test_prop_ra->spectral_snapshot (dealii::Point<1,double>(0),
    //"1d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // global refine, pol 2
  std::cout << "1D global refinement, pol 2" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->pip_set.spectral_pol_order = 2;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spectral_L2err() << std::endl;
    // test_prop_rg2->spectral_snapshot (dealii::Point<1,double>(0),
    //"1d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
}

// time-independent approach
// spectral domain is set as 2D
void static_12routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 3;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.q2_max = 1;
  test_par->grid_set.q2_min = -1;
  test_par->grid_set.q3_max = 1;
  test_par->grid_set.q3_min = -1;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = true;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "2D global refinement, pol 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<1, 2>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spectral_L2err() << std::endl;
    // test_prop_rg->spectral_snapshot (dealii::Point<1,double>(0),
    //"2d_tmp_rg_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // adaptive refine
  std::cout << "2D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "adaptive_gradient";
  auto test_prop_ra = std::make_unique<Propagator_tmp<1, 2>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spectral_L2err() << std::endl;
    // test_prop_ra->spectral_snapshot (dealii::Point<1,double>(0),
    //"3d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // global refine, pol 2
  std::cout << "2D global refinement, pol 2" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->pip_set.spectral_pol_order = 2;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<1, 2>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spectral_L2err() << std::endl;
    // test_prop_rg2->spectral_snapshot (dealii::Point<1,double>(0),
    //"3d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
}

// time-independent approach
// spectral domain is set as 3D
void static_13routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 3;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.q2_max = 1;
  test_par->grid_set.q2_min = -1;
  test_par->grid_set.q3_max = 1;
  test_par->grid_set.q3_min = -1;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = true;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "3D global refinement, pol 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<1, 3>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spectral_L2err() << std::endl;
    // test_prop_rg->spectral_snapshot (dealii::Point<1,double>(0),
    //"3d_tmp_rg_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // adaptive refine
  std::cout << "3D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->grid_set.spectral_refine_scheme = "adaptive_gradient";
  auto test_prop_ra = std::make_unique<Propagator_tmp<1, 3>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spectral_L2err() << std::endl;
    // test_prop_ra->spectral_snapshot (dealii::Point<1,double>(0),
    //"3d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
  // global refine, pol 2
  std::cout << "3D global refinement, pol 2" << std::endl;
  test_par->grid_set.spectral_max_refine_lv = 4;
  test_par->pip_set.spectral_pol_order = 2;
  test_par->grid_set.spectral_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<1, 3>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spectral_L2err() << std::endl;
    // test_prop_rg2->spectral_snapshot (dealii::Point<1,double>(0),
    //"3d_tmp_ra_lv"+dealii::Utilities::int_to_string(lv,2));
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spectral_max_refine_lv);
}

// time-dependent approach with
// fixed time-step difference
void evolve_11routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.q2_max = 1;
  test_par->grid_set.q2_min = -1;
  test_par->grid_set.q3_max = 1;
  test_par->grid_set.q3_min = -1;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = true;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  // time-dependent
  test_par->pip_set.time_dependency = true;
  test_par->pip_set.solver_scheme = 0.5;
  test_par->pip_set.physical_timediff = 0.02;
  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spectral_refine_scheme = "global";
  for (int lv = 0; lv <= 4; ++lv) {
    test_par->grid_set.spectral_min_refine_lv = lv;
    std::cout << "refine lv. " << lv << std::endl;
    for (int i = 10; i < 2100; i += 50) {
      test_par->pip_set.step_lim = i;
      test_par->pip_set.refine_cd = test_par->pip_set.step_lim + 1;
      auto test_prop_rg =
          std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
      test_prop_rg->run();
      std::cout << i << "\t" << test_prop_rg->spectral_L2err() << std::endl;
    }
  }
}

// time-dependent approach with
// fixed total evolving time
void convergence_11routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.q2_max = 1;
  test_par->grid_set.q2_min = -1;
  test_par->grid_set.q3_max = 1;
  test_par->grid_set.q3_min = -1;
  test_par->grid_set.nq1 = 3;
  test_par->grid_set.nq2 = 3;
  test_par->grid_set.nq3 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = true;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  // time-dependent
  test_par->pip_set.time_dependency = true;
  test_par->pip_set.solver_scheme = 0.5;
  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spectral_refine_scheme = "global";

  for (int lv = 0; lv <= 4; ++lv) {
    test_par->grid_set.spectral_min_refine_lv = lv;
    std::cout << "refine lv. " << lv << std::endl;
    for (int i = 5; i < 200; i += 3) {
      test_par->pip_set.step_lim = i;
      test_par->pip_set.physical_timediff = 200. / i;
      test_par->pip_set.refine_cd = test_par->pip_set.step_lim + 1;
      auto test_prop_rg =
          std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
      test_prop_rg->run();
      std::cout << i << "\t" << test_prop_rg->spectral_L2err() << std::endl;
    }
  }
}

//------------------------------------------------------------------------------
// main

int main() {
  // time-independent routines
  static_11routine();
  // static_13routine();
  // time-dependent routine with fixed steps
  evolve_11routine();
  // time-dependent routine with fixed evolving time
  convergence_11routine();
}

// END
