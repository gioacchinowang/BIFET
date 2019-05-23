// examples of a spatial advection + spatial diffusion problem which is
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
#include <deal.II/fe/fe_q.h>
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
// customized diffusion tensor

template <int spa_dim, int spe_dim>
class Diffusion_tmp final : public Diffusion<spa_dim, spe_dim> {
public:
  Diffusion_tmp(const Param *);
  virtual ~Diffusion_tmp() = default;
  Diffusion_tmp(const Diffusion_tmp &d) : Diffusion<spa_dim, spe_dim>(d) {
    this->alpha = d.alpha;
    this->beta = d.beta;
  }
  virtual Diffusion_tmp *clone() const { return new Diffusion_tmp(*this); }
  dealii::Tensor<2, spa_dim, double>
  Dxx(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &) const override;

  double alpha, beta;
};

template <int spa_dim, int spe_dim>
Diffusion_tmp<spa_dim, spe_dim>::Diffusion_tmp(const Param *) {
  this->alpha = 1.;
  this->beta = 1.;
}

// customized spatial diffusion tensor modelling
template <int spa_dim, int spe_dim>
dealii::Tensor<2, spa_dim, double> Diffusion_tmp<spa_dim, spe_dim>::Dxx(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &) const {
  dealii::Tensor<2, spa_dim, double> tmp;
  tmp = 0;
  // D[0,0]
  tmp[dealii::TableIndices<2>(0, 0)] = this->alpha;
  if (spa_dim > 1) {
    // D[1,1]
    tmp[dealii::TableIndices<2>(1, 1)] = this->beta;
    if (spa_dim > 2) {
      // D[2,2]
      tmp[dealii::TableIndices<2>(2, 2)] = this->beta;
    }
  }
  return tmp;
}

//------------------------------------------------------------------------------
// customized advection tensor

template <int spa_dim, int spe_dim>
class Advection_tmp final : public Advection<spa_dim, spe_dim> {
public:
  Advection_tmp(const Param *);
  virtual ~Advection_tmp() = default;
  Advection_tmp(const Advection_tmp &a) : Advection<spa_dim, spe_dim>(a) {
    this->az = a.az;
    this->ax = a.ax;
    this->ay = a.ay;
  }
  virtual Advection_tmp *clone() const { return new Advection_tmp(*this); }
  dealii::Tensor<1, spa_dim, double>
  Axx(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &) const override;

  double az, ax, ay;
};

template <int spa_dim, int spe_dim>
Advection_tmp<spa_dim, spe_dim>::Advection_tmp(const Param *) {
  this->az = 1.;
  this->ax = 1.;
  this->ay = 1.;
}

// spatial advection tensor modelling
template <int spa_dim, int spe_dim>
dealii::Tensor<1, spa_dim, double> Advection_tmp<spa_dim, spe_dim>::Axx(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &) const {
  dealii::Tensor<1, spa_dim, double> tmp;
  tmp = 0;
  // G[0]
  tmp[dealii::TableIndices<1>(0)] = this->az;
  if (spa_dim > 1) {
    // G[1]
    tmp[dealii::TableIndices<1>(1)] = this->ax;
    if (spa_dim > 2) {
      // G[2]
      tmp[dealii::TableIndices<1>(2)] = this->ay;
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
    this->az = s.az;
    this->ax = s.ax;
    this->ay = s.ay;
    this->L0 = s.L0;
    this->L1 = s.L1;
    this->L2 = s.L2;
    this->x0_min = s.x0_min;
    this->x1_min = s.x1_min;
    this->x2_min = s.x2_min;
  }
  virtual Source_tmp *clone() const { return new Source_tmp(*this); }
  double value(const dealii::Point<spa_dim, double> &,
               const dealii::Point<spe_dim, double> &) const override;

  double alpha, beta, az, ax, ay;
  double L0, L1, L2, x0_min, x1_min, x2_min;
};

template <int spa_dim, int spe_dim>
Source_tmp<spa_dim, spe_dim>::Source_tmp(const Param *par) {
  this->alpha = 1.;
  this->beta = 1.;
  this->az = 1.;
  this->ax = 1.;
  this->ay = 1.;
  this->x0_min = par->grid_set.x1_min;
  this->x1_min = par->grid_set.x2_min;
  this->x2_min = par->grid_set.x3_min;
  this->L0 = par->grid_set.x1_max - this->x0_min;
  this->L1 = par->grid_set.x2_max - this->x1_min;
  this->L2 = par->grid_set.x3_max - this->x2_min;
}

// customized source modelling
template <int spa_dim, int spe_dim>
double Source_tmp<spa_dim, spe_dim>::value(
    const dealii::Point<spa_dim, double> &x,
    const dealii::Point<spe_dim, double> &) const {
  const double prefact0{
      (std::exp(-this->L0) / (this->alpha + this->az) - 1. / this->az) *
      std::exp(-this->az * this->L0 / this->alpha)};
  const double dz{x[0] - this->x0_min};
  const double u0{1. / this->az +
                  prefact0 * std::exp(dz * this->az / this->alpha) -
                  std::exp(-dz) / (this->alpha + this->az)};
  const double f0{std::exp(-dz)};
  double tmp = f0;
  if (spa_dim > 1) {
    const double prefact1{
        (std::exp(-this->L1) / (this->beta + this->ax) - 1. / this->ax) *
        std::exp(-this->ax * this->L1 / this->beta)};
    const double dx{x[1] - this->x1_min};
    const double u1{1. / this->ax +
                    prefact1 * std::exp(dx * this->ax / this->beta) -
                    std::exp(-dx) / (this->beta + this->ax)};
    const double f1{std::exp(-dx)};
    tmp = tmp * u1 + f1 * u0;
    if (spa_dim > 2) {
      const double prefact2{
          (std::exp(-this->L2) / (this->beta + this->ay) - 1. / this->ay) *
          std::exp(-this->ay * this->L2 / this->beta)};
      const double dy{x[2] - this->x2_min};
      const double u2{1. / this->ay +
                      prefact2 * std::exp(dy * this->ay / this->beta) -
                      std::exp(-dy) / (this->beta + this->ay)};
      const double f2{std::exp(-dy)};
      tmp = tmp * u2 + u0 * u1 * f2;
    }
  }
  return tmp;
}

//------------------------------------------------------------------------------
// customized Simbox

// spatial frame designed for tmp
// homogeneous null boundary condition applies to upper surfaces
// lower surfaces are free from strong boundary condition
// read spatial setting paramters from Param class
template <int dim> class Frame_ADxxspatial final : public Frame<dim> {
public:
  Frame_ADxxspatial() = default;
  Frame_ADxxspatial(const Param *);
  Frame_ADxxspatial(const Frame_ADxxspatial<dim> &) = delete;
  Frame_ADxxspatial(Frame_ADxxspatial<dim> &&) = delete;
  Frame_ADxxspatial &operator=(const Frame_ADxxspatial<dim> &) = delete;
  Frame_ADxxspatial &operator=(Frame_ADxxspatial<dim> &&) = delete;
  virtual ~Frame_ADxxspatial() = default;
  void bfmap_init() override;
};

template <int dim> Frame_ADxxspatial<dim>::Frame_ADxxspatial(const Param *par) {
  // setup finite element order
  this->pol_order = par->pip_set.spatial_pol_order;
  // cache refinement limit
  this->min_refine_lv = par->grid_set.spatial_min_refine_lv;
  this->max_refine_lv = par->grid_set.spatial_max_refine_lv;
  // cache refine/coarsen ratio
  this->refine_ratio = par->grid_set.refine_ratio;
  this->coarsen_ratio = par->grid_set.coarsen_ratio;
  // setup simulation box
  switch (dim) {
  case 1:
    this->pivot_min = dealii::Point<dim, double>(par->grid_set.x1_min);
    this->pivot_max = dealii::Point<dim, double>(par->grid_set.x1_max);
    this->block_nums = {par->grid_set.nx1 - 1};
    break;
  case 2:
    this->pivot_min =
        dealii::Point<dim, double>(par->grid_set.x1_min, par->grid_set.x2_min);
    this->pivot_max =
        dealii::Point<dim, double>(par->grid_set.x1_max, par->grid_set.x2_max);
    this->block_nums = {par->grid_set.nx1 - 1, par->grid_set.nx2 - 1};
    break;
  case 3:
    this->pivot_min = dealii::Point<dim, double>(
        par->grid_set.x1_min, par->grid_set.x2_min, par->grid_set.x3_min);
    this->pivot_max = dealii::Point<dim, double>(
        par->grid_set.x1_max, par->grid_set.x2_max, par->grid_set.x3_max);
    this->block_nums = {par->grid_set.nx1 - 1, par->grid_set.nx2 - 1,
                        par->grid_set.nx3 - 1};
    break;
  default:
    assert(dim > 0 and dim < 4);
    break;
  }
  this->fe = std::make_unique<dealii::FE_Q<dim>>(this->pol_order);
}

template <int dim> void Frame_ADxxspatial<dim>::bfmap_init() {
  this->bfmap->clear();
  auto bid = this->triangulation->get_boundary_ids();
  for (auto &i : bid) {
    if (i % 2 != 0)
      this->bfmap->insert(std::make_pair(i, this->boundary.get()));
  }
}

template <int spa_dim, int spe_dim>
class Simbox_tmp final : public Simbox<spa_dim, spe_dim> {
public:
  Simbox_tmp() = default;
  Simbox_tmp(const Param *);
  virtual ~Simbox_tmp() = default;
};

template <int spa_dim, int spe_dim>
Simbox_tmp<spa_dim, spe_dim>::Simbox_tmp(const Param *par) {
  this->spatial_frame = std::make_unique<Frame_ADxxspatial<spa_dim>>(par);
  this->spectral_frame = std::make_unique<Frame_freespectral<spe_dim>>(par);
  this->sparsity = std::make_unique<dealii::SparsityPattern>();
  this->dsp = std::make_unique<dealii::DynamicSparsityPattern>();
}

//------------------------------------------------------------------------------
// customized System

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
  this->diffusion = std::make_unique<Diffusion_tmp<spa_dim, spe_dim>>(par);
  this->advection = std::make_unique<Advection_tmp<spa_dim, spe_dim>>(par);
  this->source = std::make_unique<Source_tmp<spa_dim, spe_dim>>(par);
}

template <int spa_dim, int spe_dim>
void System_tmp<spa_dim, spe_dim>::Operator::init(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const double &) {
  // auxiliary objects needed for conduct discretized integration
  auto spatial_quadrature_formula = std::make_unique<dealii::QGauss<spa_dim>>(
      simbox->spatial_frame->fe->degree + 1);
  auto spectral_quadrature_formula = std::make_unique<dealii::QGauss<spe_dim>>(
      simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  // update_gradients is critical
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_gradients |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  // fe_values in spectral domain
  // update_values is critical
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
  // allocate temporary local (per-cell) matrix holder
  auto cell_Mx =
      std::make_unique<dealii::FullMatrix<double>>(spatial_dpc, spatial_dpc);
  auto cell_Mq =
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
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (const auto &spectral_cell :
         simbox->spectral_frame->dof_handler->active_cell_iterators()) {
      spectral_fev->reinit(spectral_cell);
      // (from per-cell indeces to global indeces)
      spectral_cell->get_dof_indices(*spectral_l2g);
      // discrete integration over spectral domain per cell
      for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
           ++spectral_qid) {
        // prepare Mq
        for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
             ++alpha) {
          for (dealii::types::global_dof_index beta = 0; beta < spectral_dpc;
               ++beta) {
            cell_Mq->set(alpha, beta,
                         spectral_fev->shape_value(alpha, spectral_qid) *
                             spectral_fev->shape_value(beta, spectral_qid) *
                             spectral_fev->JxW(spectral_qid));
          } // beta
        }   // alpha
        // (clean cacher)
        system->Mq->reinit(*(simbox->spectral_frame->sparsity));
        // (convert local full matrix to global sparse matrix cacher)
        simbox->spectral_frame->constraints->distribute_local_to_global(
            *cell_Mq, *spectral_l2g, *(system->Mq));
        // discrete integration over spatial volume per cell
        for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
             ++spatial_qid) {
          const dealii::Tensor<2, spa_dim, double> diffusion_coefficient{
              system->diffusion->Dxx(
                  spatial_fev->quadrature_point(spatial_qid),
                  spectral_fev->quadrature_point(spectral_qid))};
          const dealii::Tensor<1, spa_dim, double> advection_coefficient{
              system->advection->Axx(
                  spatial_fev->quadrature_point(spatial_qid),
                  spectral_fev->quadrature_point(spectral_qid))};
          // prepare Mx
          for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
            for (dealii::types::global_dof_index j = 0; j < spatial_dpc; ++j) {
              cell_Mx->set(i, j,
                           dealii::scalar_product(
                               spatial_fev->shape_grad(i, spatial_qid),
                               diffusion_coefficient *
                                   spatial_fev->shape_grad(j, spatial_qid)) *
                                   spatial_fev->JxW(spatial_qid) -
                               dealii::scalar_product(
                                   spatial_fev->shape_grad(i, spatial_qid),
                                   advection_coefficient) *
                                   spatial_fev->shape_value(j, spatial_qid) *
                                   spatial_fev->JxW(spatial_qid));
            } // j
          }   // i
          // (clean cacher)
          system->Mx->reinit(*(simbox->spatial_frame->sparsity));
          // (convert local full matrix to global sparse matrix cacher)
          simbox->spatial_frame->constraints->distribute_local_to_global(
              *cell_Mx, *spatial_l2g, *(system->Mx));
          // accumulate to Mxq
          system->Operator_Kronecker_accumulate(simbox);
        } // spatial_qid
      }   // spectral_qid
    }     // spectral_cell loop
  }       // spatial_cell loop
}

template <int spa_dim, int spe_dim>
void System_tmp<spa_dim, spe_dim>::RHS::init(
    System<spa_dim, spe_dim> *system, const Simbox<spa_dim, spe_dim> *simbox,
    const double &) {
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
    spatial_cell->get_dof_indices(*spatial_l2g);
    for (const auto &spectral_cell :
         simbox->spectral_frame->dof_handler->active_cell_iterators()) {
      spectral_fev->reinit(spectral_cell);
      // from per-cell indeces to global indeces
      spectral_cell->get_dof_indices(*spectral_l2g);
      // discrete integration over spectral domain per cell
      for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
           ++spectral_qid) {
        // prepare spectral cell rhs
        for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
             ++alpha) {
          (*spectral_cell_rhs)[alpha] =
              spectral_fev->shape_value(alpha, spectral_qid) *
              spectral_fev->JxW(spectral_qid);
        }
        // (clean cacher)
        system->Rq->reinit(simbox->spectral_frame->dof_handler->n_dofs());
        // (convert local vector to global vector, with per-domain constraints
        // applied)
        simbox->spectral_frame->constraints->distribute_local_to_global(
            *spectral_cell_rhs, *spectral_l2g, *(system->Rq));
        // discrete integration over spatial domain per cell
        for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
             ++spatial_qid) {
          const double coefficient{system->source->value(
              spatial_fev->quadrature_point(spatial_qid),
              spectral_fev->quadrature_point(spectral_qid))};
          // prepare spatial cell rhs
          for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
            (*spatial_cell_rhs)[i] = coefficient *
                                     spatial_fev->shape_value(i, spatial_qid) *
                                     spatial_fev->JxW(spatial_qid);
          } // i
          // (clean cacher)
          system->Rx->reinit(simbox->spatial_frame->dof_handler->n_dofs());
          // (convert local vector to global vector, with per-domain constraints
          // applied)
          simbox->spatial_frame->constraints->distribute_local_to_global(
              *spatial_cell_rhs, *spatial_l2g, *(system->Rx));
          // accumulate system rhs with Kronecker product
          system->RHS_Kronecker_accumulate();
        } // spatial_qid
      }   // spectral_qid
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
  // spectral interpolation position
  dealii::Point<spe_dim, double> spectral_ref;
  // calculating L2 error w.r.t. analytical result
  double spatial_L2err() const;
  // analytical solution
  class analytical_solution : public dealii::Function<spa_dim, double> {
  public:
    analytical_solution() = default;
    analytical_solution(const Param *);
    virtual ~analytical_solution() = default;
    double value(const dealii::Point<spa_dim, double> &,
                 const unsigned int component = 0) const override;

  protected:
    double L0, L1, L2, x0_min, x1_min, x2_min;
    double alpha, beta, az, ax, ay;
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
  this->spectral_ref[0] = 0.5 * (par->grid_set.q1_max + par->grid_set.q1_min);
  if (spe_dim > 1) {
    this->spectral_ref[1] = 0.5 * (par->grid_set.q2_max + par->grid_set.q2_min);
    this->spectral_ref[2] = 0.5 * (par->grid_set.q3_max + par->grid_set.q3_min);
  }
}

template <int spa_dim, int spe_dim>
void Propagator_tmp<spa_dim, spe_dim>::solve_single_step() {
  // direct solver
  auto solver = std::make_unique<dealii::SparseDirectUMFPACK>();
  solver->initialize(*(this->system->Mxq));
  solver->vmult(*(this->solution->Snew), *(this->system->Rxq));
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
  this->alpha = 1.;
  this->beta = 1.;
  this->az = 1.;
  this->ax = 1.;
  this->ay = 1.;
  this->x0_min = par->grid_set.x1_min;
  this->x1_min = par->grid_set.x2_min;
  this->x2_min = par->grid_set.x3_min;
  this->L0 = par->grid_set.x1_max - this->x0_min;
  this->L1 = par->grid_set.x2_max - this->x1_min;
  this->L2 = par->grid_set.x3_max - this->x2_min;
}

template <int spa_dim, int spe_dim>
double Propagator_tmp<spa_dim, spe_dim>::analytical_solution::value(
    const dealii::Point<spa_dim, double> &x, const unsigned int) const {
  const double prefact0{
      (std::exp(-this->L0) / (this->alpha + this->az) - 1. / this->az) *
      std::exp(-this->az * this->L0 / this->alpha)};
  const double dz{x[0] - this->x0_min};
  double tmp{1. / this->az + prefact0 * std::exp(dz * this->az / this->alpha) -
             std::exp(-dz) / (this->alpha + this->az)};
  if (spa_dim > 1) {
    const double prefact1{
        (std::exp(-this->L1) / (this->beta + this->ax) - 1. / this->ax) *
        std::exp(-this->ax * this->L1 / this->beta)};
    const double dx{x[1] - this->x1_min};
    tmp *= (1. / this->ax + prefact1 * std::exp(dx * this->ax / this->beta) -
            std::exp(-dx) / (this->beta + this->ax));
    if (spa_dim > 2) {
      const double prefact2{
          (std::exp(-this->L2) / (this->beta + this->ay) - 1. / this->ay) *
          std::exp(-this->ay * this->L2 / this->beta)};
      const double dy{x[2] - this->x2_min};
      tmp *= (1. / this->ay + prefact2 * std::exp(dy * this->ay / this->beta) -
              std::exp(-dy) / (this->beta + this->ay));
    }
  }
  return tmp;
}

template <int spa_dim, int spe_dim>
double Propagator_tmp<spa_dim, spe_dim>::spatial_L2err() const {
  // interpolate at this->spectral_ref
  const unsigned int spectral_dofs{
      this->simbox->spectral_frame->dof_handler->n_dofs()};
  const unsigned int spatial_dofs{
      this->simbox->spatial_frame->dof_handler->n_dofs()};
  auto spa_slice = std::make_unique<dealii::Vector<double>>(spatial_dofs);
  auto tmp_spectral = std::make_unique<dealii::Vector<double>>(spectral_dofs);
  auto field_spectral =
      std::make_unique<dealii::Functions::FEFieldFunction<spe_dim>>(
          *(this->simbox->spectral_frame->dof_handler), *tmp_spectral);
  auto c_id = dealii::GridTools::find_active_cell_around_point(
      *(this->simbox->spectral_frame->dof_handler), this->spectral_ref);
  field_spectral->set_active_cell(c_id);
  for (unsigned int i = 0; i < spatial_dofs; ++i) {
    for (unsigned int j = 0; j < spectral_dofs; ++j)
      (*tmp_spectral)[j] = this->solution->new_el(i, j);
    (*spa_slice)[i] = field_spectral->value(this->spectral_ref);
  }
  // evaluate L2 err with spa_slice
  auto cellwise_err = std::make_unique<dealii::Vector<double>>(
      this->simbox->spatial_frame->triangulation->n_active_cells());
  dealii::VectorTools::integrate_difference(
      *(this->simbox->spatial_frame->dof_handler), *spa_slice,
      *(this->baseline), *cellwise_err,
      dealii::QIterated<spa_dim>(dealii::QTrapez<1>(),
                                 this->simbox->spatial_frame->fe->degree + 2),
      dealii::VectorTools::L2_norm);
  return dealii::VectorTools::compute_global_error(
      *(this->simbox->spatial_frame->triangulation), *cellwise_err,
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
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.nq1 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spatial_L2err() << std::endl;
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // adaptive refine
  std::cout << "1D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "adaptive_kelly";
  auto test_prop_ra = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spatial_L2err() << std::endl;
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // global refine, pol 2
  std::cout << "1D global refinement, pol 2" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->pip_set.spatial_pol_order = 2;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spatial_L2err() << std::endl;
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
}

// time-independent approach
// spectral domain is set as 2D
void static_21routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 2;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.nq1 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "2D global refinement, pol 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<2, 1>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spatial_L2err() << std::endl;
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // adaptive refine
  std::cout << "2D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "adaptive_kelly";
  auto test_prop_ra = std::make_unique<Propagator_tmp<2, 1>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spatial_L2err() << std::endl;
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // global refine, pol 2
  std::cout << "2D global refinement, pol 2" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->pip_set.spatial_pol_order = 2;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<2, 1>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spatial_L2err() << std::endl;
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
}

// time-independent approach
// spectral domain is set as 3D
void static_31routine() {
  // parameter holder
  auto test_par = std::make_unique<Param>();
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial grid
  test_par->pip_set.spatial_dim = 3;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 1.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.nq1 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;

  unsigned int lv;
  // global refine
  std::cout << "3D global refinement, pol 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg = std::make_unique<Propagator_tmp<3, 1>>(test_par.get());
  test_prop_rg->init();
  lv = 0;
  do {
    test_prop_rg->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg->spatial_L2err() << std::endl;
    test_prop_rg->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // adaptive refine
  std::cout << "3D adaptive refinement, pol. 1" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->grid_set.spatial_refine_scheme = "adaptive_kelly";
  auto test_prop_ra = std::make_unique<Propagator_tmp<3, 1>>(test_par.get());
  test_prop_ra->init();
  lv = 0;
  do {
    test_prop_ra->solve_single_step();
    std::cout << lv << "\t" << test_prop_ra->spatial_L2err() << std::endl;
    test_prop_ra->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
  // global refine, pol 2
  std::cout << "3D global refinement, pol 2" << std::endl;
  test_par->grid_set.spatial_max_refine_lv = 4;
  test_par->pip_set.spatial_pol_order = 2;
  test_par->grid_set.spatial_refine_scheme = "global";
  auto test_prop_rg2 = std::make_unique<Propagator_tmp<3, 1>>(test_par.get());
  test_prop_rg2->init();
  lv = 0;
  do {
    test_prop_rg2->solve_single_step();
    std::cout << lv << "\t" << test_prop_rg2->spatial_L2err() << std::endl;
    test_prop_rg2->refine();
    lv++;
  } while (lv <= test_par->grid_set.spatial_max_refine_lv);
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
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.nq1 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  // time-dependent
  test_par->pip_set.time_dependency = true;
  test_par->pip_set.solver_scheme = 0.5;
  test_par->pip_set.physical_timediff = 0.002;

  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spatial_refine_scheme = "global";
  for (int lv = 0; lv <= 4; ++lv) {
    test_par->grid_set.spatial_min_refine_lv = lv;
    std::cout << "refine lv. " << lv << std::endl;
    for (int i = 10; i < 2100; i += 50) {
      test_par->pip_set.step_lim = i;
      test_par->pip_set.refine_cd = test_par->pip_set.step_lim + 1;
      auto test_prop_rg =
          std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
      test_prop_rg->run();
      std::cout << i << "\t" << test_prop_rg->spatial_L2err() << std::endl;
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
  test_par->grid_set.x2_max = 1.;
  test_par->grid_set.x2_min = 0.;
  test_par->grid_set.x3_max = 1.;
  test_par->grid_set.x3_min = 0.;
  test_par->grid_set.nx1 = 3;
  test_par->grid_set.nx2 = 3;
  test_par->grid_set.nx3 = 3;
  // spectral grid
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 1;
  test_par->grid_set.q1_min = -1;
  test_par->grid_set.nq1 = 3;
  // grid refine limits
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.do_spatial_refine = true;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  // global refine
  std::cout << "1D global refinement, pol 1" << std::endl;
  test_par->grid_set.spatial_refine_scheme = "global";
  // time-dependent
  test_par->pip_set.time_dependency = true;
  test_par->pip_set.solver_scheme = 0.5;

  for (int lv = 0; lv <= 4; ++lv) {
    test_par->grid_set.spatial_min_refine_lv = lv;
    std::cout << "refine lv. " << lv << std::endl;
    for (int i = 5; i < 300; i += 5) {
      test_par->pip_set.step_lim = i;
      test_par->pip_set.physical_timediff = 6. / i;
      test_par->pip_set.refine_cd = test_par->pip_set.step_lim + 1;
      auto test_prop_rg =
          std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
      test_prop_rg->run();
      std::cout << i << "\t" << test_prop_rg->spatial_L2err() << std::endl;
    }
  }
}

//------------------------------------------------------------------------------
// main

int main() {
  // time-independent routines
  static_11routine();
  // static_21routine();
  // static_31routine();
  // time-dependent routine with fixed steps
  evolve_11routine();
  // time-dependent routine with fixed evolving time
  convergence_11routine();
}

// END
