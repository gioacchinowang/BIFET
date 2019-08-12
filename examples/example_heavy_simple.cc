// in this example we solve a simple CR electron transport scenario with
// spatial diffusion + energy loss in R^{1+1} dimension setting
// both spatial and spectral domains are treated as reduced from 3D spherical
// symmetric domain this assumption brings in extra geometric tensors in System
// assembling although derived temporary classes are built with dimension
// templates the math within works only for spherical symmetric domain reduced
// in 1D both spatial diffusion and energy loss tensors are energy dependent
// spatial frame is bounded by zero value on all surfaces
// spectral frame is bounded by zero on upper boundary surfaces (the upper
// energy limit) the system is built with implicit convention for physical
// quantities
// - energy in GeV
// - spatial position/distance in kpc
// - time in Gyr

#include <cassert>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#ifndef NTIMING
#include <timer.h>
#endif

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <advection.h>
#include <diffusion.h>
#include <frame.h>
#include <param.h>
#include <propagator.h>
#include <simbox.h>
#include <source.h>
#include <system.h>

//------------------------------------------------------------------------------
// customized diffusion

template <int spa_dim, int spe_dim>
class Diffusion_tmp final : public Diffusion<spa_dim, spe_dim> {
public:
  Diffusion_tmp() = default;
  virtual ~Diffusion_tmp() = default;
  Diffusion_tmp(const Diffusion_tmp &d) : Diffusion<spa_dim, spe_dim>(d) {}
  virtual Diffusion_tmp *clone() const { return new Diffusion_tmp(*this); }
  dealii::Tensor<2, spa_dim, double>
  Dxx(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &) const override;
};

// isotropic spatial diffusion with power law energy dependency
template <int spa_dim, int spe_dim>
dealii::Tensor<2, spa_dim, double> Diffusion_tmp<spa_dim, spe_dim>::Dxx(
    const dealii::Point<spa_dim, double> &x,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<2, spa_dim, double> tmp;
  tmp = 0;
  double Esph{std::exp(q[0])};
  double Xsph2{x[0] * x[0]};
  // x^2 for spherical symmetric
  const double iso_dxx = 63.0 * std::pow(Esph, 0.33) * Xsph2;
  for (unsigned int i = 0; i < spa_dim; ++i)
    tmp[dealii::TableIndices<2>(i, i)] = iso_dxx;
  return tmp;
}

//------------------------------------------------------------------------------
// customized advection

template <int spa_dim, int spe_dim>
class Advection_tmp final : public Advection<spa_dim, spe_dim> {
public:
  Advection_tmp() = default;
  virtual ~Advection_tmp() = default;
  Advection_tmp(const Advection_tmp &a) : Advection<spa_dim, spe_dim>(a) {}
  virtual Advection_tmp *clone() const { return new Advection_tmp(*this); }
  dealii::Tensor<1, spe_dim, double>
  Aqq(const dealii::Point<spa_dim, double> &,
      const dealii::Point<spe_dim, double> &) const override;

protected:
  dealii::Tensor<1, spe_dim, double>
  ic_loss(const dealii::Point<spa_dim, double> &,
          const dealii::Point<spe_dim, double> &) const;
  dealii::Tensor<1, spe_dim, double>
  sync_loss(const dealii::Point<spa_dim, double> &,
            const dealii::Point<spe_dim, double> &) const;
  dealii::Tensor<1, spe_dim, double>
  coul_loss(const dealii::Point<spa_dim, double> &,
            const dealii::Point<spe_dim, double> &) const;
  dealii::Tensor<1, spe_dim, double>
  brem_loss(const dealii::Point<spa_dim, double> &,
            const dealii::Point<spe_dim, double> &) const;
};

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Advection_tmp<spa_dim, spe_dim>::Aqq(
    const dealii::Point<spa_dim, double> &x,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  double Esph2{std::exp(q[0]) * std::exp(q[0])};
  tmp += this->ic_loss(x, q);
  tmp += this->sync_loss(x, q);
  tmp += this->coul_loss(x, q);
  tmp += this->brem_loss(x, q);
  // -1. as loss
  return -Esph2 * tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Advection_tmp<spa_dim, spe_dim>::ic_loss(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  double Esph{std::exp(q[0])};
  const double iso_aqq = 0.83 * Esph * Esph;
  for (unsigned int i = 0; i < spe_dim; ++i)
    tmp[dealii::TableIndices<1>(i)] = iso_aqq;
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Advection_tmp<spa_dim, spe_dim>::sync_loss(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  double Esph{std::exp(q[0])};
  const double iso_aqq = 1.99 * Esph * Esph;
  for (unsigned int i = 0; i < spe_dim; ++i)
    tmp[dealii::TableIndices<1>(i)] = iso_aqq;
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Advection_tmp<spa_dim, spe_dim>::coul_loss(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  double Esph{std::exp(q[0])};
  const double iso_aqq = 14.18 + 0.96 * std::log(Esph);
  for (unsigned int i = 0; i < spe_dim; ++i)
    tmp[dealii::TableIndices<1>(i)] = iso_aqq;
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> Advection_tmp<spa_dim, spe_dim>::brem_loss(
    const dealii::Point<spa_dim, double> &,
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  double Esph{std::exp(q[0])};
  const double iso_aqq = 40.0 * Esph;
  for (unsigned int i = 0; i < spe_dim; ++i)
    tmp[dealii::TableIndices<1>(i)] = iso_aqq;
  return tmp;
}

//------------------------------------------------------------------------------
// customized source

template <int spa_dim, int spe_dim>
class Source_tmp final : public Source<spa_dim, spe_dim> {
public:
  Source_tmp() = default;
  virtual ~Source_tmp() = default;
  Source_tmp(const Source_tmp &s) : Source<spa_dim, spe_dim>(s) {}
  virtual Source_tmp *clone() const { return new Source_tmp(*this); }
  double value(const dealii::Point<spa_dim, double> &,
               const dealii::Point<spe_dim, double> &) const override;
};

template <int spa_dim, int spe_dim>
double Source_tmp<spa_dim, spe_dim>::value(
    const dealii::Point<spa_dim, double> &x,
    const dealii::Point<spe_dim, double> &q) const {
  double Esph{std::exp(q[0])};

  // SNR source
  return 0.39 * std::exp(-x[0] / 0.5) * std::pow(Esph, -2.2);

  // Burkert source
  // return std::pow(Esph,-2.2)*0.0156/std::pow((x[0]+0.5)*(x[0]*x[0]+0.25),2);

  // NFW source
  // if (x[0]<0.01)
  //    return std::pow(Esph,-2.2)/std::pow(0.01*1.01*1.01,2);
  // else
  //    return std::pow(Esph,-2.2)/std::pow(x[0]*(1.+x[0])*(1.+x[0]),2);
}

//------------------------------------------------------------------------------
// customized frame

template <int dim> class Frame__tmp final : public Frame<dim> {
public:
  Frame__tmp() = default;
  Frame__tmp(const Param *);
  virtual ~Frame__tmp() = default;
  void bfmap_init() override;
};

template <int dim> Frame__tmp<dim>::Frame__tmp(const Param *par) {
  this->pol_order = par->pip_set.spatial_pol_order;
  this->min_refine_lv = par->grid_set.spatial_min_refine_lv;
  this->max_refine_lv = par->grid_set.spatial_max_refine_lv;
  this->refine_ratio = par->grid_set.refine_ratio;
  this->coarsen_ratio = par->grid_set.coarsen_ratio;
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

template <int dim> void Frame__tmp<dim>::bfmap_init() {
  this->bfmap->clear();
  auto bid = this->triangulation->get_boundary_ids();
  for (auto &i : bid) {
    if (i % 2 != 0)
      this->bfmap->insert(std::make_pair(i, this->boundary.get()));
  }
}

//------------------------------------------------------------------------------
// customised solution

template <int spa_dim, int spe_dim>
class Solution_tmp final : public Solution<spa_dim, spe_dim> {
public:
  Solution_tmp(const Param *);
  virtual ~Solution_tmp() = default;
  Solution_tmp(const Solution_tmp &s) : Solution<spa_dim, spe_dim>(s) {}
  virtual Solution_tmp *clone() const { return new Solution_tmp(*this); }
  class Spatial_initial : public Solution<spa_dim, spe_dim>::Spatial_initial {
  public:
    using dealii::Function<1, double>::operator=;
    Spatial_initial() = default;
    Spatial_initial(const Spatial_initial &s)
        : Solution<spa_dim, spe_dim>::Spatial_initial(s) {
      *this = s;
    }
    virtual Spatial_initial *clone() const {
      return new Spatial_initial(*this);
    }
    virtual ~Spatial_initial() = default;
    double value(const dealii::Point<spa_dim, double> &,
                 const unsigned int component = 0) const override;
  };
  class Spectral_initial : public Solution<spa_dim, spe_dim>::Spectral_initial {
  public:
    using dealii::Function<1, double>::operator=;
    Spectral_initial() = default;
    Spectral_initial(const Spectral_initial &s)
        : Solution<spa_dim, spe_dim>::Spectral_initial(s) {
      *this = s;
    }
    virtual Spectral_initial *clone() const {
      return new Spectral_initial(*this);
    }
    virtual ~Spectral_initial() = default;
    double value(const dealii::Point<spe_dim, double> &,
                 const unsigned int component = 0) const override;
  };
};

template <int spa_dim, int spe_dim>
Solution_tmp<spa_dim, spe_dim>::Solution_tmp(const Param *) {
  this->Snew = std::make_unique<dealii::Vector<double>>();
  this->Sold = std::make_unique<dealii::Vector<double>>();
  this->spatial = std::make_unique<
      typename Solution_tmp<spa_dim, spe_dim>::Spatial_initial>();
  this->spectral = std::make_unique<
      typename Solution_tmp<spa_dim, spe_dim>::Spectral_initial>();
}

template <int spa_dim, int spe_dim>
double Solution_tmp<spa_dim, spe_dim>::Spatial_initial::value(
    const dealii::Point<spa_dim, double> &, const unsigned int) const {
  return 0.;
}

template <int spa_dim, int spe_dim>
double Solution_tmp<spa_dim, spe_dim>::Spectral_initial::value(
    const dealii::Point<spe_dim, double> &, const unsigned int) const {
  return 0.;
}

//------------------------------------------------------------------------------
// customized simbox

template <int spa_dim, int spe_dim>
class Simbox_tmp final : public Simbox<spa_dim, spe_dim> {
public:
  Simbox_tmp() = default;
  Simbox_tmp(const Param *);
  virtual ~Simbox_tmp() = default;
};

template <int spa_dim, int spe_dim>
Simbox_tmp<spa_dim, spe_dim>::Simbox_tmp(const Param *par) {
  this->spatial_frame = std::make_unique<Frame__tmp<spa_dim>>(par);
  this->spectral_frame = std::make_unique<dGFrame_spectral<spe_dim>>(par);
  this->sparsity = std::make_unique<dealii::SparsityPattern>();
  this->dsp = std::make_unique<dealii::DynamicSparsityPattern>();
}

//------------------------------------------------------------------------------
// customized system

template <int spa_dim, int spe_dim>
class System_tmp final : public System<spa_dim, spe_dim> {
public:
  System_tmp();
  virtual ~System_tmp() = default;
  System_tmp(const System_tmp<spa_dim, spe_dim> &s)
      : System<spa_dim, spe_dim>(s) {}
  virtual System_tmp *clone() const { return new System_tmp(*this); }
  class Operator : public System<spa_dim, spe_dim>::Operator {
  public:
    Operator() = default;
    virtual ~Operator() = default;
    virtual Operator *clone() const { return new Operator(*this); }
    void init(System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
              const double &) override;

  protected:
    dealii::Tensor<2, spa_dim, double>
    Tx(const dealii::Point<spa_dim, double> &) const;
    dealii::Tensor<1, spa_dim, double>
    dTx(const dealii::Point<spa_dim, double> &) const;
    dealii::Tensor<2, spe_dim, double>
    Tq(const dealii::Point<spe_dim, double> &) const;
    dealii::Tensor<1, spe_dim, double>
    dTq(const dealii::Point<spe_dim, double> &) const;
    // assemble face terms for init function
    void assemble_spectral_face(
        System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
        const dealii::FEValues<spa_dim> *, const unsigned int &,
        const dealii::FEFaceValuesBase<spe_dim> *,
        const dealii::FEFaceValuesBase<spe_dim> *,
        std::vector<dealii::types::global_dof_index> *,
        std::vector<dealii::types::global_dof_index> *,
        dealii::FullMatrix<double> *, dealii::FullMatrix<double> *,
        dealii::FullMatrix<double> *, dealii::FullMatrix<double> *);
  };
  class RHS : public System<spa_dim, spe_dim>::RHS {
  public:
    RHS() = default;
    virtual ~RHS() = default;
    virtual RHS *clone() const { return new RHS(*this); }
    void init(System<spa_dim, spe_dim> *, const Simbox<spa_dim, spe_dim> *,
              const double &) override;
  };
};

template <int spa_dim, int spe_dim> System_tmp<spa_dim, spe_dim>::System_tmp() {
  this->Mx = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Mxq = std::make_unique<dealii::SparseMatrix<double>>();
  this->Rxq = std::make_unique<dealii::Vector<double>>();
  this->Rx = std::make_unique<dealii::Vector<double>>();
  this->Rq = std::make_unique<dealii::Vector<double>>();
  this->op =
      std::make_unique<typename System_tmp<spa_dim, spe_dim>::Operator>();
  this->rhs = std::make_unique<typename System_tmp<spa_dim, spe_dim>::RHS>();
  this->diffusion = std::make_unique<Diffusion_tmp<spa_dim, spe_dim>>();
  this->advection = std::make_unique<Advection_tmp<spa_dim, spe_dim>>();
  this->source = std::make_unique<Source_tmp<spa_dim, spe_dim>>();
}

template <int spa_dim, int spe_dim>
dealii::Tensor<2, spa_dim, double> System_tmp<spa_dim, spe_dim>::Operator::Tx(
    const dealii::Point<spa_dim, double> &x) const {
  dealii::Tensor<2, spa_dim, double> tmp;
  tmp = 0;
  if (x[0] == 0)
    return tmp; // avoid singularity
  tmp[dealii::TableIndices<2>(0, 0)] = 1. / (x[0] * x[0]);
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spa_dim, double> System_tmp<spa_dim, spe_dim>::Operator::dTx(
    const dealii::Point<spa_dim, double> &x) const {
  dealii::Tensor<1, spa_dim, double> tmp;
  tmp = 0;
  if (x[0] == 0)
    return tmp; // avoid singularity
  tmp[dealii::TableIndices<1>(0)] = -2. / (x[0] * x[0] * x[0]);
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<2, spe_dim, double> System_tmp<spa_dim, spe_dim>::Operator::Tq(
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<2, spe_dim, double> tmp;
  tmp = 0;
  tmp[dealii::TableIndices<2>(0, 0)] = std::exp(-3. * q[0]);
  return tmp;
}

template <int spa_dim, int spe_dim>
dealii::Tensor<1, spe_dim, double> System_tmp<spa_dim, spe_dim>::Operator::dTq(
    const dealii::Point<spe_dim, double> &q) const {
  dealii::Tensor<1, spe_dim, double> tmp;
  tmp = 0;
  tmp[dealii::TableIndices<1>(0)] = -3. * std::exp(-3. * q[0]);
  return tmp;
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
  auto spectral_face_quadrature_formula =
      std::make_unique<dealii::QGauss<spe_dim - 1>>(
          simbox->spectral_frame->fe->degree + 1);
  // fe_values in spatial domain
  auto spatial_fev = std::make_unique<dealii::FEValues<spa_dim>>(
      *(simbox->spatial_frame->fe), *spatial_quadrature_formula,
      dealii::update_values | dealii::update_gradients |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  // fe_values in spectral domain
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
#ifndef NDEBUG
      assert(spectral_cell->active());
#endif
      spectral_cell->get_dof_indices(*spectral_l2g);
      // DIFFUSION OPERATOR
      for (unsigned int spatial_qid = 0; spatial_qid < spatial_q_points;
           ++spatial_qid) {
        // discrete integration over spectral domain per cell
        for (unsigned int spectral_qid = 0; spectral_qid < spectral_q_points;
             ++spectral_qid) {
          // prepare Mx (for diffusion)
          for (dealii::types::global_dof_index i = 0; i < spatial_dpc; ++i) {
            for (dealii::types::global_dof_index j = 0; j < spatial_dpc; ++j) {
              const dealii::Tensor<2, spa_dim, double> diffusion_coefficient{
                  system->diffusion->Dxx(
                      spatial_fev->quadrature_point(spatial_qid),
                      spectral_fev->quadrature_point(spectral_qid))};
              const dealii::Tensor<2, spa_dim, double> geometric_coefficient{
                  this->Tx(spatial_fev->quadrature_point(spatial_qid))};
              const dealii::Tensor<1, spa_dim, double>
                  geometric_div_coefficient{
                      this->dTx(spatial_fev->quadrature_point(spatial_qid))};
              cell_Mx->set(
                  i, j,
                  (dealii::scalar_product(
                       geometric_coefficient *
                           spatial_fev->shape_grad(i, spatial_qid),
                       diffusion_coefficient *
                           spatial_fev->shape_grad(j, spatial_qid)) +
                   spatial_fev->shape_value(i, spatial_qid) *
                       dealii::scalar_product(
                           geometric_div_coefficient,
                           diffusion_coefficient *
                               spatial_fev->shape_grad(j, spatial_qid))) *
                      spatial_fev->JxW(spatial_qid));
            } // j
          }   // i
          // (convert local full matrix to global sparse matrix cacher)
          system->Mx->reinit(*(simbox->spatial_frame->sparsity));
          simbox->spatial_frame->constraints->distribute_local_to_global(
              *cell_Mx, *spatial_l2g, *(system->Mx));
          // prepare Mq (for diffusion)
          for (dealii::types::global_dof_index alpha = 0; alpha < spectral_dpc;
               ++alpha) {
            for (dealii::types::global_dof_index beta = 0; beta < spectral_dpc;
                 ++beta) {
              cell_Mq_uivi->set(
                  alpha, beta,
                  spectral_fev->shape_value(alpha, spectral_qid) *
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
      }   // spatial_qid
      // ENERGY LOSS (ADVECTION) OPERATOR
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
                  spectral_fev->quadrature_point(spectral_qid))};
          const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
              this->Tq(spectral_fev->quadrature_point(spectral_qid))};
          const dealii::Tensor<1, spe_dim, double> geometric_div_coefficient{
              this->dTq(spectral_fev->quadrature_point(spectral_qid))};
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
                        spectral_fefv->quadrature_point(spectral_qid))};
                const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
                    this->Tq(spectral_fefv->quadrature_point(spectral_qid))};
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
                    cell_Mq_ueve.get());
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
              // refinement level and no face has children anx2more
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
                    cell_Mq_ueve.get());
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
    dealii::FullMatrix<double> *uive, dealii::FullMatrix<double> *ueve) {
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
                               fefv->quadrature_point(spectral_qid))};
    const dealii::Tensor<2, spe_dim, double> geometric_coefficient{
        this->Tq(fefv->quadrature_point(spectral_qid))};
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
              spectral_fev->quadrature_point(spectral_qid))};
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
// customized propagator
template <int spa_dim, int spe_dim>
class Propagator_tmp final : public Propagator<spa_dim, spe_dim> {
public:
  Propagator_tmp(const Param *);
  virtual ~Propagator_tmp() = default;
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
  this->solution = std::make_unique<Solution_tmp<spa_dim, spe_dim>>(par);
  this->system = std::make_unique<System_tmp<spa_dim, spe_dim>>();
}

// main (testing) routines
int main() {
  auto test_par = std::make_unique<Param>();
  // solver precision
  test_par->pip_set.iteration = 1000;
  test_par->pip_set.tolerance = 1.e-12;
  // spatial domain
  test_par->pip_set.spatial_dim = 1;
  test_par->pip_set.spatial_pol_order = 1;
  test_par->grid_set.x1_max = 30.;
  test_par->grid_set.x1_min = 0.;
  test_par->grid_set.nx1 = 100;
  // spectral domain
  test_par->pip_set.spectral_dim = 1;
  test_par->pip_set.spectral_pol_order = 1;
  test_par->grid_set.q1_max = 7;
  test_par->grid_set.q1_min = -5;
  test_par->grid_set.nq1 = 600;
  // grid refine limits
  test_par->grid_set.do_spatial_refine = false;
  test_par->grid_set.do_spectral_refine = false;
  test_par->grid_set.spatial_refine_scheme = "adaptive_kelly";
  test_par->grid_set.spectral_refine_scheme = "adaptive_kelly";
  test_par->grid_set.spectral_min_refine_lv = 0;
  test_par->grid_set.spatial_min_refine_lv = 0;
  test_par->grid_set.spatial_max_refine_lv = 0;
  test_par->grid_set.spectral_max_refine_lv = 0;
  test_par->grid_set.refine_ratio = 0.5;
  test_par->grid_set.coarsen_ratio = 0;
  // time-dependency
  test_par->pip_set.time_dependency = false;
  // for time-independent solving, set step_lim as 0
  // otherwise the single step solver will repeat
  test_par->pip_set.step_lim = 0;
  test_par->pip_set.refine_cd = test_par->pip_set.step_lim + 1;
  test_par->pip_set.solver_scheme = 0.5;
  test_par->pip_set.evo_lim = 1.e-3;
  test_par->pip_set.physical_timediff = 1.e-2;

  auto test_prop = std::make_unique<Propagator_tmp<1, 1>>(test_par.get());
  test_prop->run();

  // get density
  // test_prop->density_snapshot ("example_cre_simple");
  // get spectral snapshot
  test_prop->spectral_snapshot(dealii::Point<1, double>(0),
                               "example_cre_simple_snr_q0");
  test_prop->spectral_snapshot(dealii::Point<1, double>(2),
                               "example_cre_simple_snr_q2");
  test_prop->spectral_snapshot(dealii::Point<1, double>(4),
                               "example_cre_simple_snr_q4");
  test_prop->spectral_snapshot(dealii::Point<1, double>(8),
                               "example_cre_simple_snr_q8");
  // get spatial snapshot
  test_prop->spatial_snapshot(dealii::Point<1, double>(-2),
                              "example_cre_simple_snr_xm2");
  test_prop->spatial_snapshot(dealii::Point<1, double>(0),
                              "example_cre_simple_snr_x0");
  test_prop->spatial_snapshot(dealii::Point<1, double>(2),
                              "example_cre_simple_snr_x2");
  test_prop->spatial_snapshot(dealii::Point<1, double>(4),
                              "example_cre_simple_snr_x4");
}

// END
